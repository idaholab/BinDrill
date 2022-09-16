# coding: utf-8
from sqlalchemy import Column, ForeignKey, Index, Integer, LargeBinary, Text, text, Float
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata

class DiscoEmbedding(Base):
    __tablename__ = 'embeddings'

    id = Column(Integer, primary_key=True)
    function = Column(Text)
    executable = Column(Text)
    data_set = Column(Text)
    address = Column(Integer)
    embedding = Column(Text)
