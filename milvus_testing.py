from pymilvus import list_collections, connections, FieldSchema, Collection, CollectionSchema, DataType, utility
from analysis.models.sqlalchemy_models import Embedding, Mapping, Base, Function
from analysis.models.disco_models import DiscoEmbedding

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, joinedload, sessionmaker
import numpy as np
import logging
import torch.nn as nn
import torch
from tqdm import tqdm
from inspect import getmembers
from time import perf_counter, time
from sqlalchemy.pool import NullPool
from functools import wraps


def timing(func):
    """
    Timing decrorator function. stix2.exceptions.ExtraPropertiesError
    """

    @wraps(func)
    def wrap(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"func:{func.__name__} took: {end - start:2.5f} sec")
        return result

    return wrap



class MilvusConnector:
    def __init__(self, host='127.0.0.1', port='19530', sqlite_path=None, collection_name='joint_embeddings', mapping_file=None, connect=True):
        if connect:
            self.connected = True
            if mapping_file is not None:
                self.mapping_engine = create_engine(f'sqlite:///{mapping_file}', poolclass=NullPool)
                self.mapping_sess = sessionmaker(self.mapping_engine)
                Base.metadata.create_all(self.mapping_engine, tables=[Mapping.__table__])
            self.mapping_to_str = {}
            self.mapping_to_int = {}
            # self.load_mappings()

            self.coll_name = collection_name
            connections.add_connection(default={"host": host, "port": port})
            connections.connect(alias='default')

            self.search_params = {"ef": 1024, "nprobe" : 512}

            self.index_params = {
            "metric_type":"IP",
            "index_type":"HNSW",
            "params":{"M":64, "efConstruction":512}
            }

            if collection_name not in self.get_collections():
                self.prepare()
            else:
                self.collection = Collection(name=self.coll_name)
            if sqlite_path is not None:
                self.set_input_sqlite_db(sqlite_path)
        else:
            self.connected = False

    # NOTE: need to generalize in the base class
    @timing
    def format_embeddings(self, db_embeds):
        vectors = ['joint']
        attrs = ['dataset', 'executable', 'function', 'address', 'version', 
                'arch', 'blocks', 'bytes',  'asm', 'hlil', 'joint']

        l = [[] for attr in attrs]
        for db_embed in tqdm(db_embeds, leave=False):
            for index, attr in enumerate(attrs):
                if attr in vectors:
                    value = getattr(db_embed, attr)
                    l[index].append(self.text_to_list(value))
                else:
                    value = getattr(db_embed.function_ref, attr)
                    if isinstance(value, str):
                        l[index].append(self.remap_to_int(value))
                    else:
                        l[index].append(value)

        # self.mapping_sess.commit()
        return l

    def set_input_sqlite_db(self, sqlite_path):
        
        self.input_engine = create_engine(f'sqlite:///{sqlite_path}',  poolclass=NullPool)
        self.input_session = sessionmaker(self.input_engine)

    def get_embedding(self):
        with self.input_session() as session:
            return self.session.query(Embedding).options(joinedload(Embedding.function_ref)).first()

    def get_embeddings(self, executable=None, dataset=None, limit=None, offset=None, include_bin=None, exclude_bin=None, func_filter=None):  
        with self.input_session() as session:
            query = session.query(Embedding).join(Function)
            query = query.options(joinedload(Embedding.function_ref))
            if dataset is not None:
                query = query.filter(Embedding.function_ref.has(dataset=dataset))
            if executable is not None:
                query = query.filter(Embedding.function_ref.has(executable=executable))
            if offset is not None:
                query = query.offset(offset)
            if exclude_bin is not None:
                query = query.filter(Embedding.function_ref.has(Function.executable!=exclude_bin))
            if func_filter is not None and func_filter != '':
                if func_filter.startswith('!'):
                    func_filter = func_filter[1:]
                    query = query.filter(Embedding.function_ref.has(Function.dataset!=func_filter))
                else:
                    query = query.filter(Embedding.function_ref.has(Function.dataset==func_filter))
            if include_bin is not None:
                query = query.filter(Embedding.function_ref.has(Function.executable==include_bin))
            query = query.order_by(Function.address)
            if limit is not None:
                query = query.limit(limit)
            
            return query.all()

    def getbins(self):
        with self.input_session() as session:
            query = session.query(Function.executable).distinct(Function.executable)
            return query.all()

    def push_embeddings(self, sqlite_path):
        pass

    def get_num_entities(self):
       return self.collection.num_entities

    @timing
    def search(self, embedding, field="embedding", limit=5, expr=None, output_fields=None):
        search_params = {"metric_type": self.index_params["metric_type"], "params": self.search_params}
        self.wait_until_ready()
        return self.collection.search(data=embedding, anns_field=field, param=search_params, limit=limit, expr=expr, output_fields=output_fields)
       
    def wait_until_ready(self):
        utility.wait_for_index_building_complete(self.coll_name)
        self.load()
        utility.wait_for_loading_complete(self.coll_name)

    def prepare(self):
        dim = 256
        id_field = FieldSchema(
            name="id", dtype=DataType.INT64, description="primary_field")
        dataset_field = FieldSchema(
            name="dataset", dtype=DataType.INT64, description="dataset_field")
        executable_field = FieldSchema(
            name="executable", dtype=DataType.INT64, description="executable_field")
        function_field = FieldSchema(
            name="function", dtype=DataType.INT64, description="function_field")
        address_field = FieldSchema(
            name="address", dtype=DataType.INT64, description="address_field")
        version_field = FieldSchema(
            name="version", dtype=DataType.INT64, description="version_field")
        arch_field = FieldSchema(
            name="arch", dtype=DataType.INT64, description="arch_field")
        blocks_field = FieldSchema(
            name="blocks", dtype=DataType.INT64, description="blocks_field")
        bytes_field = FieldSchema(
            name="bytes", dtype=DataType.INT64, description="bytes_field")
        asm_field = FieldSchema(
            name="asm", dtype=DataType.INT64, description="asm_field")
        hlil_field = FieldSchema(
            name="hlil", dtype=DataType.INT64, description="hlil_field")
        embedding_field = FieldSchema(
            name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        schema = CollectionSchema(fields=[id_field, dataset_field, executable_field, function_field,
                                          address_field, version_field, arch_field, blocks_field, bytes_field,
                                           asm_field, hlil_field, embedding_field], primary_field='id', auto_id=True,
                                  description='embedding collection')
        self.collection = Collection(name=self.coll_name, schema=schema)

        self.collection.create_index(field_name="embedding", index_params=self.index_params)

    def check_create_paritions(self, partion_name):
        if partion_name not in self.collection.partitions():
            self.collection.create_partition(partion_name)

    def get_collections(self):
        return list_collections()
    
    def print_collection_details(self):
        for col_name in self.get_collections():
            col = Collection(name=col_name)
            print(col)
            print(f'Num Entities: {col.num_entities}')
            print(f'Indexes: {col.indexes}')
            print(f'Partitions: {col.partitions}')

    def insert(self, l):
        self.collection.insert(l)

    def reset(self):
        self.collection.drop()

    def load_mappings(self):
        with self.mapping_sess() as session:
            for mapping in session.query(Mapping).all():
                self.mapping_to_int[mapping.key] = mapping.id
                self.mapping_to_str[mapping.id] = mapping.key
    
    def remap_to_str(self, elem):
        if elem in self.mapping_to_str:
            return self.mapping_to_str[elem]
        else:
            with self.mapping_sess() as session:
                mapping = session.query(Mapping).filter(Mapping.id == elem).first()
                if mapping is not None:
                    self.mapping_to_int[mapping.key] = elem
                    self.mapping_to_str[elem] = mapping.key
                    return mapping.key

    def remap_to_int(self, elem):
        if elem in self.mapping_to_int:
            return self.mapping_to_int[elem]
        else:
            with self.mapping_sess() as session:
                mapping = session.query(Mapping).filter(Mapping.key == elem).first()
                if mapping is not None:
                    self.mapping_to_int[elem] = mapping.id
                    self.mapping_to_str[mapping.id] = elem
                    return mapping.id

                else:
                    mapping = Mapping(key=elem)
                    session.add(mapping)
                    session.commit()
                    self.mapping_to_int[elem] = mapping.id
                    self.mapping_to_str[mapping.id] = elem
                    return mapping.id

    @staticmethod
    def text_to_torch(e: str) -> np.array:
        e = e.replace('\n', '')
        e = e.replace('[', '')
        e = e.replace(']', '')
        emb = np.fromstring(e, dtype=float, sep=' ')
        return torch.from_numpy(emb)

    @staticmethod
    def text_to_list(e: str) -> list:
        raw_emb = MilvusConnector.text_to_torch(e)
        raw_emb = nn.functional.normalize(raw_emb, p=2, dim=-1)
        return raw_emb.tolist()

    def load(self):
        self.collection.load()

    def unload(self):
        self.collection.release()



class DiscoMilvusConnector(MilvusConnector):
    # NOTE: will need to fix
    def get_embedding(self):
        with self.input_session() as session:
            return self.session.query(Embedding).options(joinedload(Embedding.function_ref)).first()

    # NOTE: will need to fix
    def get_embeddings(self, executable=None, dataset=None, limit=None, offset=None, include_bin=None, exclude_bin=None, func_filter=None):
        with self.input_session() as session:
            query = session.query(DiscoEmbedding)
            if dataset is not None:
                query = query.filter(DiscoEmbedding.dataset==dataset)
            if executable is not None:
                query = query.filter(DiscoEmbedding.executable==executable)
            if exclude_bin is not None:
                query = query.filter(DiscoEmbedding.executable!=exclude_bin)
            if func_filter is not None and func_filter != '':
                if func_filter.startswith('!'):
                    func_filter = func_filter[1:]
                    query = query.filter(DiscoEmbedding.dataset!=dataset)
                else:
                   query = query.filter(DiscoEmbedding.dataset==dataset)
            if include_bin is not None:
                query = query.filter(DiscoEmbedding.executable==include_bin)
            query = query.order_by(DiscoEmbedding.address)
            if limit is not None:
                query = query.limit(limit)
            return query.all()

    # NOTE: will need to fix
    def getbins(self):
        with self.input_session() as session:
            query = session.query(DiscoEmbedding.executable).distinct(DiscoEmbedding.executable)
            return query.all()

    # NOTE: need to generalize in the base class
    def prepare(self):
        dim = 256
        id_field = FieldSchema(
            name="id", dtype=DataType.INT64, description="primary_field")
        dataset_field = FieldSchema(
            name="dataset", dtype=DataType.INT64, description="dataset_field")
        executable_field = FieldSchema(
            name="executable", dtype=DataType.INT64, description="executable_field")
        function_field = FieldSchema(
            name="function", dtype=DataType.INT64, description="function_field")
        address_field = FieldSchema(
            name="address", dtype=DataType.INT64, description="address_field")
        version_field = FieldSchema(
            name="version", dtype=DataType.INT64, description="version_field")
        arch_field = FieldSchema(
            name="arch", dtype=DataType.INT64, description="arch_field")
        blocks_field = FieldSchema(
            name="blocks", dtype=DataType.INT64, description="blocks_field")
        bytes_field = FieldSchema(
            name="bytes", dtype=DataType.INT64, description="bytes_field")
        asm_field = FieldSchema(
            name="asm", dtype=DataType.INT64, description="asm_field")
        hlil_field = FieldSchema(
            name="hlil", dtype=DataType.INT64, description="hlil_field")
        embedding_field = FieldSchema(
            name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        schema = CollectionSchema(fields=[id_field, dataset_field, executable_field, function_field,
                                          address_field, version_field, arch_field, blocks_field, bytes_field,
                                           asm_field, hlil_field, embedding_field], primary_field='id', auto_id=True,
                                  description='embedding collection')
        self.collection = Collection(name=self.coll_name, schema=schema)

        self.collection.create_index(field_name="embedding", index_params=self.index_params)


  




if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    input_embedding_db = 'sqlite/all.sqlite'
    mapping_db = 'mapping.sqlite'
    new_params = []

    # for x in [32768, 65536]:
    #     for y in [1, 8, 32, 1024, 4096]:
    #         if y > x:
    #             continue
    #         new_params.append(({"metric_type":"IP",
    #         "index_type":"IVF_FLAT",
    #         "params":{"nlist": x}},
    #         {"nprobe" : y}))

    #         new_params.append(({"metric_type":"IP",
    #         "index_type":"IVF_SQ8",
    #         "params":{"nlist": x}},
    #         {"nprobe" : y}))

    # for x in [48, 64]:
    #     for y in [8, 32, 64, 512, 1024, 4096]:
    #         new_params.append(({"metric_type":"IP",
    #         "index_type":"HNSW",
    #         "params":{"M": x, "efConstruction":512}},
    #         {"ef" : y}))    

    # l = []

    

    # MC = MilvusConnector(mapping_file = '')
    # MC.set_input_sqlite_db(input_embedding_db)
    # embeddings = MC.get_embeddings()
    # formatted_embeddings = MC.format_embeddings(embeddings)
    # embed = MC.input_session.query(Embedding).first()
            
    # embed = MilvusConnector.text_to_list(embed.joint)

    # embeddings = MC.get_embeddings(executable="4098b54c9d27b00ce34d04ffac24213ed28993a2854827851b157d63407c2e4e.exe")
    # one_bin_embeds = [MilvusConnector.text_to_list(embedding.joint) for embedding in embeddings]
