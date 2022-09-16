# coding: utf-8
from sqlalchemy import Column, ForeignKey, Index, Integer, LargeBinary, Text, text, Float, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

CompareBase = declarative_base()
CompareMetadata = CompareBase.metadata


class Function(CompareBase):
    id = Column(Integer, primary_key=True)
    __tablename__ = 'function'
    func_name = Column(Text)
    address = Column(Integer, index=True)
    byte_size = Column(Integer, index=True)
    joint_embed = Column(Text)
    hlil_embed = Column(Text)
    disco_embed = Column(Text)
    main = Column(Boolean, index=True)
    asm = Column(Text)
    llil = Column(Text)
    mlil = Column(Text)
    hlil = Column(Text)
    vex = Column(Text)
    exe_id = Column(Integer, ForeignKey('exe.id'), index=True)
    exe_rel = relationship('Executable')


class Match(CompareBase):
    id = Column(Integer, primary_key=True)
    __tablename__ = 'match'
    score = Column(Float, index=True)
    score_type = Column(Text)
    best = Column(Boolean, index=True)

    src_id = Column(Integer, ForeignKey('function.id'), index=True)
    dst_id = Column(Integer, ForeignKey('function.id'), index=True)
    src_rel = relationship('Function', foreign_keys=src_id)
    dst_rel = relationship('Function', foreign_keys=dst_id)

class Executable(CompareBase):
    id = Column(Integer, primary_key=True, index=True)
    __tablename__ = 'exe'
    exe_name = Column(Text, index=True)
    bin_path = Column(Text, nullable=False)
    exe_hash = Column(Text, index=True)
    category = Column(Text, index=True)
    version = Column(Text)
    compiler = Column(Text)
    options = Column(Text)
    arch = Column(Text, index=True)
    blocks = Column(Integer)
    _bytes = Column('bytes', Integer)
    dataset = Column(Text, nullable=False, index=True)

# class ExeCompared(CompareBase):
#     id = Column(Integer, primary_key=True, index=True)
#     __tablename__ = 'exe'
#     src_exe_id =
#     dst_exe_id = 
#     score_type = Column(Text)


