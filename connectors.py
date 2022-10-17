from sqlalchemy.pool import NullPool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, joinedload, load_only
from models import Function, Embedding
import numpy as np
import torch
import torch.nn as nn
import logging
from OrientDBWrapper import RestAPIWrapper
from pprint import pprint
import json

# from bulk_compare import CompareGraph
from milvus_testing import MilvusConnector, DiscoMilvusConnector, timing


def text_to_np(e: str) -> np.array:
    e = e.replace("\n", "")
    e = e.replace("[", "")
    e = e.replace("]", "")
    emb = np.fromstring(e, dtype=float, sep=" ")
    return emb

        
def return_details(embed, function_list, score=None):
    attribs = [
        "function",
        "executable",
        "dataset",
        "address",
        "bytes",
        "asm",
        "llil",
        "mlil",
        "hlil",
    ]

    d = {attrib: getattr(function_list[embed], attrib) for attrib in attribs}
    if score is not None:
        d["score"] = score
    return d


class EmbedSearchConnector:
    id_prefix = "base"
    def __init__(self, path=None, port=None, host=None, config=None, **kwargs):
        if config is not None:
            self.milvus_port =  config['milvus_port']
            self.milvus_host = config['milvus_host']
            self.orientdb_host = config['orientdb']['host']
            self.orientdb_port = config['orientdb']['port']
            self.orientdb_user = config['orientdb']['user']
            self.orientdb_pass = config['orientdb']['pass']
            self.mapping_file = config['mapping_file']
            self.function_groups_file = config['function_groups_file']
            self.limit = config['func_limit']

    def regen_id(self):
        _id = f'''{self.id_prefix}{"con"     
        if self.MC is not None and self.MC  
        else ""}{self.sqlite_path       
        if self.sqlite_path is not None  
        else ""}'''
        self.id = _id

    @classmethod
    def gen_id(cls, MC=None, sqlite_path=None):
        _id = f'''{cls.id_prefix}{"con"     
        if MC is not None and MC  
        else ""}{sqlite_path       
        if sqlite_path is not None  
        else ""}'''
        return _id

class TroglodyteSearchConnector(EmbedSearchConnector):
    id_prefix = "trogbase"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.MC = None

    def getfuncs(self, func_filter=None, **kwargs):
        # return self.MC.get_embeddings(func_filter=func_filter, **kwargs)
        return self.MC.get_embeddings(func_filter=func_filter, **kwargs)

    def getbins(self):
        return self.MC.getbins()

    def set_input_sqlite(self, sqlite_path):
        self.sqlite_path = sqlite_path
        self.MC.set_input_sqlite_db(sqlite_path)
        self.regen_id()

class SqliteTroglodyteSearchConnector(TroglodyteSearchConnector):
    id_prefix = "sqlitetrog"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "func_filter" in kwargs:
            self.func_filter = kwargs["func_filter"]
        else:
            self.func_filter = None
        self.output_fields = [
            "dataset",
            "executable",
            "function",
            "address",
            "version",
            "arch",
            "blocks",
            "bytes",
            "asm",
            "hlil",
        ]
        self.direct_fields = ["address", "bytes", "blocks"]
        self.id = None
        
        self.sqlite_path = None



    def connect(
        self, *args, sqlite_file=None, connect=True, **kwargs,
    ):
        if connect:
            
            self.MC_compare = MilvusConnector(connect=False, host=self.milvus_host, port=self.milvus_port)
            self.MC_compare.set_input_sqlite_db(sqlite_file)
            self.regen_id()
        else:
            self.MC = MilvusConnector(connect=False, host=self.milvus_host, port=self.milvus_port)
        self.regen_id()
        

    @timing
    def search(self, bin_embeddings, *args, topk=5, func_filter=None, **kwargs):
        bin_embeddings_full = sorted(bin_embeddings, key=lambda x: x.function_ref.address)
        compare_embeddings_full = self.MC_compare.get_embeddings(func_filter=func_filter, **kwargs)
        bin_embeddings = [
            MilvusConnector.text_to_torch(embedding.joint)
            for embedding in bin_embeddings_full
        ]
        compare_embeddings = [
            MilvusConnector.text_to_torch(embedding.joint)
            for embedding in compare_embeddings_full
        ]
        bin_embeddings_stacked = torch.stack(bin_embeddings)
        compare_embeddings_stacked =  torch.stack(compare_embeddings)

        bin_embeddings_n = nn.functional.normalize(bin_embeddings_stacked, p=2, dim=-1)
        compare_embeddings_n = nn.functional.normalize(compare_embeddings_stacked, p=2, dim=-1)
        dot_similarity = bin_embeddings_n @ compare_embeddings_n.T
        topk_embeds = torch.topk(dot_similarity, 5, dim=-1)
        l = []
        for index, row in enumerate(topk_embeds[0]):
            k=[]
            for index2, score in enumerate(row):
                act_index = topk_embeds[1][index][index2]
                # print(compare_embeddings_full[index2].function_ref.function, act_index)
                #this is the problem
                d = self.format_func_data(compare_embeddings_full[act_index])
                d["score"] = score.item()
                k.append(d)
            l.append((self.format_func_data(bin_embeddings_full[index]), k))
        return l
        
    def format_func_data(self, db_embed):

        d = {}
        for index, attr in enumerate(self.output_fields):
            if hasattr(db_embed.function_ref, attr):
                value = getattr(db_embed.function_ref, attr)
                d[attr] = value
            else:
                value = getattr(db_embed, attr)
                d[attr] = value

        return d

class SqliteDiscoSearchConnector(EmbedSearchConnector):
    id_prefix = "sqlitedisco"
    def __init__(self, *args, path=None, orientdb_db=None, **kwargs):
        super().__init__(*args, **kwargs)
        if "func_filter" in kwargs:
            self.func_filter = kwargs["func_filter"]
        else:
            self.func_filter = None
        self.output_fields = [
            "data_set",
            "executable",
            "function",
            "address",
            "bytes",
            "vex"
        ]
        if orientdb_db is not None:
            try:
                self.orient_con = RestAPIWrapper(db=orientdb_db, host=self.orientdb_host, port=self.orientdb_port, user=self.orientdb_user, password=self.orientdb_pass)
            except Exception as e:
                self.orient_con = None
                logging.exception(e)
        self.direct_fields = ["address", "bytes", "blocks"]
        self.id = None
        self.MC = None
        self.sqlite_path = None



    def connect(
        self, *args, sqlite_file=None, connect=True, **kwargs,
    ):
        if connect:
            self.MC_compare = DiscoMilvusConnector(connect=False, host=self.milvus_host, port=self.milvus_port)
            self.MC_compare.set_input_sqlite_db(sqlite_file)
            self.regen_id()
        else:
            self.MC = DiscoMilvusConnector(connect=False, host=self.milvus_host, port=self.milvus_port)
        self.regen_id()

    def set_input_sqlite(self, sqlite_path):
        self.sqlite_path = sqlite_path
        self.MC.set_input_sqlite_db(sqlite_path)
        self.regen_id()


    def getbins(self):
        return self.MC.getbins()

    def getfuncs(self, func_filter=None, **kwargs):
        funcs = self.MC.get_embeddings(func_filter=func_filter, **kwargs)
        return self.add_orient_data(funcs)

    def add_orient_data(self, funcs):
        for func in funcs:
            results = self.orient_con.query(f"""select normalized_instructions from block where function.name="{func.function}" and library.name="{func.executable}" order by block_id""", limit=1)
            vex = json.loads(results['result'][0]['normalized_instructions'])
            vex_flat = [v for vex_list in vex for v in vex_list]
            vex_text = "\n".join(vex_flat)
            func.vex = vex_text
            # func.bytes = len(vex_flat)
        return funcs

    @timing
    def search(self, bin_embeddings, *args, exclude_bin=None, topk=5, func_filter=None, **kwargs):
        bin_embeddings_full = sorted(bin_embeddings, key=lambda x: x.address)
        compare_embeddings_full = self.MC_compare.get_embeddings(func_filter=func_filter, exclude_bin=exclude_bin, **kwargs)
        self.compare_embeddings_full = self.add_orient_data(compare_embeddings_full)
        bin_embeddings = [
            DiscoMilvusConnector.text_to_torch(embedding.embedding)
            for embedding in bin_embeddings_full
        ]
        compare_embeddings = [
            DiscoMilvusConnector.text_to_torch(embedding.embedding)
            for embedding in compare_embeddings_full
        ]
        bin_embeddings_stacked = torch.stack(bin_embeddings)
        compare_embeddings_stacked =  torch.stack(compare_embeddings)

        bin_embeddings_n = nn.functional.normalize(bin_embeddings_stacked, p=2, dim=-1)
        compare_embeddings_n = nn.functional.normalize(compare_embeddings_stacked, p=2, dim=-1)
        dot_similarity = bin_embeddings_n @ compare_embeddings_n.T
        topk_embeds = torch.topk(dot_similarity, 5, dim=-1)
        l = []
        for index, row in enumerate(topk_embeds[0]):
            k=[]
            prev = None
            for index2, score in enumerate(row):
                act_index = topk_embeds[1][index][index2]
                d = self.format_func_data(compare_embeddings_full[index2])
                d["score"] = score.item() if score.item() < 1 else 1.0
               
                k.append(d)

            if not hasattr(bin_embeddings_full[index], "bytes"):
                if index != len(topk_embeds[0]) -1 :
                    bytes_ = bin_embeddings_full[index+1].address - bin_embeddings_full[index].address
                else:
                    bytes_ = 10
            else:
                bytes_ = None
            l.append((self.format_func_data(bin_embeddings_full[index], bytes_=bytes_), k))
        return l

    def format_func_data(self, db_embed, bytes_ = None):
        d = {}
        for index, attr in enumerate(self.output_fields):
            if hasattr(db_embed, attr):
                value = getattr(db_embed, attr)
                d[attr] = value
        if bytes_ is not None:
            d["bytes"] = bytes_

        return d


class MilvusTroglodyteSearchConnector(TroglodyteSearchConnector):
    id_prefix = "milvustrog"
    def __init__(self, *args, path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if "func_filter" in kwargs:
            self.func_filter = kwargs["func_filter"]
        else:
            self.func_filter = None
        self.output_fields = [
            "dataset",
            "executable",
            "function",
            "address",
            "version",
            "arch",
            "blocks",
            "bytes",
            "asm",
            "hlil",
        ]
        self.direct_fields = ["address", "bytes", "blocks"]
        self.id = None
        self.sqlite_path = None



    def connect(
        self, connect=True, host="127.0.0.1", port="19530", mapping_file="mapping.sqlite"
    ):
        if connect:
            self.MC = MilvusConnector(
                host=self.milvus_host, 
                port=self.milvus_port, 
                mapping_file=self.mapping_file,
                collection_name="joint_embeddings",
            )
        else:
            self.MC = MilvusConnector(connect=False, host=self.milvus_host, port=self.milvus_port)
        self.regen_id()




    @timing
    def search(self, bin_embeddings, exclude_bin=None, topk=5, func_filter=None):
        bin_embeddings = sorted(bin_embeddings, key=lambda x: x.function_ref.address)
        milvus_bin_embeddings = [
            MilvusConnector.text_to_list(embedding.joint)
            for embedding in bin_embeddings
        ]
        expr = (
            (f"dataset == {str(self.MC.remap_to_int(func_filter))}" if not func_filter.startswith('!') else f"dataset != {str(self.MC.remap_to_int(func_filter[1:]))}")
            if (func_filter is not None and func_filter != "")
            else None
        )
        if exclude_bin is not None:
            if expr is not None:
                expr = f"executable != {self.MC.remap_to_int(exclude_bin)} && "+ expr
            else:
                expr = f"executable != {self.MC.remap_to_int(exclude_bin)}"

        results = self.MC.search(
            milvus_bin_embeddings,
            output_fields=self.output_fields,
            limit=topk,
            expr=expr,
        )
        l = []
        for index, subresults in enumerate(results):
            k = []
            for result in subresults:
                d = {
                    attrib: self.MC.remap_to_str(getattr(result.entity, attrib))
                    for attrib in self.output_fields
                }
                d.update(
                    {
                        attrib: getattr(result.entity, attrib)
                        for attrib in self.direct_fields
                    }
                )
                d["score"] = result.score
                k.append(d)

            l.append((self.format_func_data(bin_embeddings[index]), k))
        return l

    def format_func_data(self, db_embed):

        d = {}
        for index, attr in enumerate(self.output_fields):
            if hasattr(db_embed.function_ref, attr):
                value = getattr(db_embed.function_ref, attr)
                d[attr] = value
            else:
                value = getattr(db_embed, attr)
                d[attr] = value

        return d


# class CompareGraphSearchConnector(EmbedSearchConnector):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.cg = CompareGraph(compare_db=self.path)
#         if "func_filter" in kwargs:
#             self.func_filter = kwargs["func_filter"]
#         else:
#             self.func_filter = None

#     def getbins(self):
#         return self.cg.getbins()

#     def getfuncs(self, *args, **kwargs):
#         return self.cg.getfuncs(*args, func_filter=self.func_filter, **kwargs)

#     def search(self, *args, **kwargs):
#         return self.cg.search(*args, **kwargs)
