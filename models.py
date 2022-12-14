# coding: utf-8
from sqlalchemy import Column, ForeignKey, Index, Integer, LargeBinary, Text, text, Float
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata

class Mapping(Base):
    __tablename__ = 'mapping'
    id = Column(Integer, primary_key=True, index=True)
    key = Column(Text, unique=True, index=True)

class Comparision(Base):
    id = Column(Integer, primary_key=True)
    __tablename__ = 'comparision'
    src_bin = Column(Text)
    dst_bin = Column(Text)
    src_dataset = Column(Text)
    dst_dataset = Column(Text)
    score = Column(Float)
    score_type = Column(Text)

class Function(Base):
    __tablename__ = 'functions'
    __table_args__ = (
        Index('executable_function_index', 'executable', 'function'),
        Index('functions_dataset_arch_compiler_options_version_executable_function_address_uindex', 'dataset', 'arch', 'compiler', 'options', 'version', 'executable', 'function', 'address', unique=True),
        Index('function_address_index', 'function', 'address')
    )

    id = Column(Integer, primary_key=True)
    dataset = Column(Text, nullable=False, index=True)
    bin_path = Column(Text, nullable=False)
    bnb_path = Column(Text)
    executable = Column(Text, nullable=False, index=True)
    function = Column(Text, nullable=False, index=True)
    address = Column(Integer, index=True)
    version = Column(Text)
    compiler = Column(Text)
    options = Column(Text)
    arch = Column(Text)
    blocks = Column(Integer)
    bytes = Column(Integer)
    embedding = Column(Text)
    asm = Column(Text)
    llil = Column(Text)
    mlil = Column(Text)
    hlil = Column(Text)
    vex = Column(Text)


class Embedding(Base):
    __tablename__ = 'embeddings'

    function_id = Column(ForeignKey('functions.id'), primary_key=True)
    function_ref = relationship("Function", uselist=False)
    vex = Column(Text)
    llil = Column(Text)
    mlil = Column(Text)
    hlil = Column(Text)
    topo = Column(Text)
    ast = Column(Text)
    cfg = Column(Text)
    dfg = Column(Text)
    joint = Column(Text)
    umap = Column(Text)
    pacmap = Column(Text)
    tsne = Column(Text)


class HlilFeature(Base):
    __tablename__ = 'hlil_feature'

    function_id = Column(ForeignKey('functions.id'), primary_key=True)
    blocks = Column(Integer, server_default=text("0"))
    bytes = Column(Integer, server_default=text("0"))
    HLIL_NOP = Column(Integer, server_default=text("0"))
    HLIL_BLOCK = Column(Integer, server_default=text("0"))
    HLIL_IF = Column(Integer, server_default=text("0"))
    HLIL_WHILE = Column(Integer, server_default=text("0"))
    HLIL_DO_WHILE = Column(Integer, server_default=text("0"))
    HLIL_FOR = Column(Integer, server_default=text("0"))
    HLIL_SWITCH = Column(Integer, server_default=text("0"))
    HLIL_CASE = Column(Integer, server_default=text("0"))
    HLIL_BREAK = Column(Integer, server_default=text("0"))
    HLIL_CONTINUE = Column(Integer, server_default=text("0"))
    HLIL_JUMP = Column(Integer, server_default=text("0"))
    HLIL_RET = Column(Integer, server_default=text("0"))
    HLIL_NORET = Column(Integer, server_default=text("0"))
    HLIL_GOTO = Column(Integer, server_default=text("0"))
    HLIL_LABEL = Column(Integer, server_default=text("0"))
    HLIL_VAR_DECLARE = Column(Integer, server_default=text("0"))
    HLIL_VAR_INIT = Column(Integer, server_default=text("0"))
    HLIL_ASSIGN = Column(Integer, server_default=text("0"))
    HLIL_ASSIGN_UNPACK = Column(Integer, server_default=text("0"))
    HLIL_VAR = Column(Integer, server_default=text("0"))
    HLIL_STRUCT_FIELD = Column(Integer, server_default=text("0"))
    HLIL_ARRAY_INDEX = Column(Integer, server_default=text("0"))
    HLIL_SPLIT = Column(Integer, server_default=text("0"))
    HLIL_DEREF = Column(Integer, server_default=text("0"))
    HLIL_DEREF_FIELD = Column(Integer, server_default=text("0"))
    HLIL_ADDRESS_OF = Column(Integer, server_default=text("0"))
    HLIL_CONST = Column(Integer, server_default=text("0"))
    HLIL_CONST_PTR = Column(Integer, server_default=text("0"))
    HLIL_EXTERN_PTR = Column(Integer, server_default=text("0"))
    HLIL_FLOAT_CONST = Column(Integer, server_default=text("0"))
    HLIL_IMPORT = Column(Integer, server_default=text("0"))
    HLIL_ADD = Column(Integer, server_default=text("0"))
    HLIL_ADC = Column(Integer, server_default=text("0"))
    HLIL_SUB = Column(Integer, server_default=text("0"))
    HLIL_SBB = Column(Integer, server_default=text("0"))
    HLIL_AND = Column(Integer, server_default=text("0"))
    HLIL_OR = Column(Integer, server_default=text("0"))
    HLIL_XOR = Column(Integer, server_default=text("0"))
    HLIL_LSL = Column(Integer, server_default=text("0"))
    HLIL_LSR = Column(Integer, server_default=text("0"))
    HLIL_ASR = Column(Integer, server_default=text("0"))
    HLIL_ROL = Column(Integer, server_default=text("0"))
    HLIL_RLC = Column(Integer, server_default=text("0"))
    HLIL_ROR = Column(Integer, server_default=text("0"))
    HLIL_RRC = Column(Integer, server_default=text("0"))
    HLIL_MUL = Column(Integer, server_default=text("0"))
    HLIL_MULU_DP = Column(Integer, server_default=text("0"))
    HLIL_MULS_DP = Column(Integer, server_default=text("0"))
    HLIL_DIVU = Column(Integer, server_default=text("0"))
    HLIL_DIVU_DP = Column(Integer, server_default=text("0"))
    HLIL_DIVS = Column(Integer, server_default=text("0"))
    HLIL_DIVS_DP = Column(Integer, server_default=text("0"))
    HLIL_MODU = Column(Integer, server_default=text("0"))
    HLIL_MODU_DP = Column(Integer, server_default=text("0"))
    HLIL_MODS = Column(Integer, server_default=text("0"))
    HLIL_MODS_DP = Column(Integer, server_default=text("0"))
    HLIL_NEG = Column(Integer, server_default=text("0"))
    HLIL_NOT = Column(Integer, server_default=text("0"))
    HLIL_SX = Column(Integer, server_default=text("0"))
    HLIL_ZX = Column(Integer, server_default=text("0"))
    HLIL_LOW_PART = Column(Integer, server_default=text("0"))
    HLIL_CALL = Column(Integer, server_default=text("0"))
    HLIL_CMP_E = Column(Integer, server_default=text("0"))
    HLIL_CMP_NE = Column(Integer, server_default=text("0"))
    HLIL_CMP_SLT = Column(Integer, server_default=text("0"))
    HLIL_CMP_ULT = Column(Integer, server_default=text("0"))
    HLIL_CMP_SLE = Column(Integer, server_default=text("0"))
    HLIL_CMP_ULE = Column(Integer, server_default=text("0"))
    HLIL_CMP_SGE = Column(Integer, server_default=text("0"))
    HLIL_CMP_UGE = Column(Integer, server_default=text("0"))
    HLIL_CMP_SGT = Column(Integer, server_default=text("0"))
    HLIL_CMP_UGT = Column(Integer, server_default=text("0"))
    HLIL_TEST_BIT = Column(Integer, server_default=text("0"))
    HLIL_BOOL_TO_INT = Column(Integer, server_default=text("0"))
    HLIL_ADD_OVERFLOW = Column(Integer, server_default=text("0"))
    HLIL_SYSCALL = Column(Integer, server_default=text("0"))
    HLIL_TAILCALL = Column(Integer, server_default=text("0"))
    HLIL_INTRINSIC = Column(Integer, server_default=text("0"))
    HLIL_BP = Column(Integer, server_default=text("0"))
    HLIL_TRAP = Column(Integer, server_default=text("0"))
    HLIL_UNDEF = Column(Integer, server_default=text("0"))
    HLIL_UNIMPL = Column(Integer, server_default=text("0"))
    HLIL_UNIMPL_MEM = Column(Integer, server_default=text("0"))
    HLIL_FADD = Column(Integer, server_default=text("0"))
    HLIL_FSUB = Column(Integer, server_default=text("0"))
    HLIL_FMUL = Column(Integer, server_default=text("0"))
    HLIL_FDIV = Column(Integer, server_default=text("0"))
    HLIL_FSQRT = Column(Integer, server_default=text("0"))
    HLIL_FNEG = Column(Integer, server_default=text("0"))
    HLIL_FABS = Column(Integer, server_default=text("0"))
    HLIL_FLOAT_TO_INT = Column(Integer, server_default=text("0"))
    HLIL_INT_TO_FLOAT = Column(Integer, server_default=text("0"))
    HLIL_FLOAT_CONV = Column(Integer, server_default=text("0"))
    HLIL_ROUND_TO_INT = Column(Integer, server_default=text("0"))
    HLIL_FLOOR = Column(Integer, server_default=text("0"))
    HLIL_CEIL = Column(Integer, server_default=text("0"))
    HLIL_FTRUNC = Column(Integer, server_default=text("0"))
    HLIL_FCMP_E = Column(Integer, server_default=text("0"))
    HLIL_FCMP_NE = Column(Integer, server_default=text("0"))
    HLIL_FCMP_LT = Column(Integer, server_default=text("0"))
    HLIL_FCMP_LE = Column(Integer, server_default=text("0"))
    HLIL_FCMP_GE = Column(Integer, server_default=text("0"))
    HLIL_FCMP_GT = Column(Integer, server_default=text("0"))
    HLIL_FCMP_O = Column(Integer, server_default=text("0"))
    HLIL_FCMP_UO = Column(Integer, server_default=text("0"))
    HLIL_WHILE_SSA = Column(Integer, server_default=text("0"))
    HLIL_DO_WHILE_SSA = Column(Integer, server_default=text("0"))
    HLIL_FOR_SSA = Column(Integer, server_default=text("0"))
    HLIL_VAR_INIT_SSA = Column(Integer, server_default=text("0"))
    HLIL_ASSIGN_MEM_SSA = Column(Integer, server_default=text("0"))
    HLIL_ASSIGN_UNPACK_MEM_SSA = Column(Integer, server_default=text("0"))
    HLIL_VAR_SSA = Column(Integer, server_default=text("0"))
    HLIL_ARRAY_INDEX_SSA = Column(Integer, server_default=text("0"))
    HLIL_DEREF_SSA = Column(Integer, server_default=text("0"))
    HLIL_DEREF_FIELD_SSA = Column(Integer, server_default=text("0"))
    HLIL_CALL_SSA = Column(Integer, server_default=text("0"))
    HLIL_SYSCALL_SSA = Column(Integer, server_default=text("0"))
    HLIL_INTRINSIC_SSA = Column(Integer, server_default=text("0"))
    HLIL_VAR_PHI = Column(Integer, server_default=text("0"))
    HLIL_MEM_PHI = Column(Integer, server_default=text("0"))


class HlilGraph(Base):
    __tablename__ = 'hlil_graph'

    function_id = Column(ForeignKey('functions.id'), primary_key=True)
    cfg = Column(LargeBinary)
    ast = Column(LargeBinary)
    dfg = Column(LargeBinary)


class LlilFeature(Base):
    __tablename__ = 'llil_feature'

    function_id = Column(ForeignKey('functions.id'), primary_key=True, unique=True)
    blocks = Column(Integer, server_default=text("0"))
    bytes = Column(Integer, server_default=text("0"))
    LLIL_ADC = Column(Integer, server_default=text("0"))
    LLIL_ADD = Column(Integer, server_default=text("0"))
    LLIL_ADD_OVERFLOW = Column(Integer, server_default=text("0"))
    LLIL_AND = Column(Integer, server_default=text("0"))
    LLIL_ASR = Column(Integer, server_default=text("0"))
    LLIL_BOOL_TO_INT = Column(Integer, server_default=text("0"))
    LLIL_BP = Column(Integer, server_default=text("0"))
    LLIL_CALL = Column(Integer, server_default=text("0"))
    LLIL_CALL_PARAM = Column(Integer, server_default=text("0"))
    LLIL_CALL_STACK_ADJUST = Column(Integer, server_default=text("0"))
    LLIL_CEIL = Column(Integer, server_default=text("0"))
    LLIL_CMP_E = Column(Integer, server_default=text("0"))
    LLIL_CMP_NE = Column(Integer, server_default=text("0"))
    LLIL_CMP_SGE = Column(Integer, server_default=text("0"))
    LLIL_CMP_SGT = Column(Integer, server_default=text("0"))
    LLIL_CMP_SLE = Column(Integer, server_default=text("0"))
    LLIL_CMP_SLT = Column(Integer, server_default=text("0"))
    LLIL_CMP_UGE = Column(Integer, server_default=text("0"))
    LLIL_CMP_UGT = Column(Integer, server_default=text("0"))
    LLIL_CMP_ULE = Column(Integer, server_default=text("0"))
    LLIL_CMP_ULT = Column(Integer, server_default=text("0"))
    LLIL_CONST = Column(Integer, server_default=text("0"))
    LLIL_CONST_PTR = Column(Integer, server_default=text("0"))
    LLIL_DIVS = Column(Integer, server_default=text("0"))
    LLIL_DIVS_DP = Column(Integer, server_default=text("0"))
    LLIL_DIVU = Column(Integer, server_default=text("0"))
    LLIL_DIVU_DP = Column(Integer, server_default=text("0"))
    LLIL_EXTERN_PTR = Column(Integer, server_default=text("0"))
    LLIL_FABS = Column(Integer, server_default=text("0"))
    LLIL_FADD = Column(Integer, server_default=text("0"))
    LLIL_FCMP_E = Column(Integer, server_default=text("0"))
    LLIL_FCMP_GE = Column(Integer, server_default=text("0"))
    LLIL_FCMP_GT = Column(Integer, server_default=text("0"))
    LLIL_FCMP_LE = Column(Integer, server_default=text("0"))
    LLIL_FCMP_LT = Column(Integer, server_default=text("0"))
    LLIL_FCMP_NE = Column(Integer, server_default=text("0"))
    LLIL_FCMP_O = Column(Integer, server_default=text("0"))
    LLIL_FCMP_UO = Column(Integer, server_default=text("0"))
    LLIL_FDIV = Column(Integer, server_default=text("0"))
    LLIL_FLAG = Column(Integer, server_default=text("0"))
    LLIL_FLAG_BIT = Column(Integer, server_default=text("0"))
    LLIL_FLAG_COND = Column(Integer, server_default=text("0"))
    LLIL_FLAG_GROUP = Column(Integer, server_default=text("0"))
    LLIL_FLOAT_CONST = Column(Integer, server_default=text("0"))
    LLIL_FLOAT_CONV = Column(Integer, server_default=text("0"))
    LLIL_FLOAT_TO_INT = Column(Integer, server_default=text("0"))
    LLIL_FLOOR = Column(Integer, server_default=text("0"))
    LLIL_FMUL = Column(Integer, server_default=text("0"))
    LLIL_FNEG = Column(Integer, server_default=text("0"))
    LLIL_FSQRT = Column(Integer, server_default=text("0"))
    LLIL_FSUB = Column(Integer, server_default=text("0"))
    LLIL_FTRUNC = Column(Integer, server_default=text("0"))
    LLIL_GOTO = Column(Integer, server_default=text("0"))
    LLIL_IF = Column(Integer, server_default=text("0"))
    LLIL_INT_TO_FLOAT = Column(Integer, server_default=text("0"))
    LLIL_INTRINSIC = Column(Integer, server_default=text("0"))
    LLIL_JUMP = Column(Integer, server_default=text("0"))
    LLIL_JUMP_TO = Column(Integer, server_default=text("0"))
    LLIL_LOAD = Column(Integer, server_default=text("0"))
    LLIL_LOW_PART = Column(Integer, server_default=text("0"))
    LLIL_LSL = Column(Integer, server_default=text("0"))
    LLIL_LSR = Column(Integer, server_default=text("0"))
    LLIL_MEM_PHI = Column(Integer, server_default=text("0"))
    LLIL_MODS = Column(Integer, server_default=text("0"))
    LLIL_MODS_DP = Column(Integer, server_default=text("0"))
    LLIL_MODU = Column(Integer, server_default=text("0"))
    LLIL_MODU_DP = Column(Integer, server_default=text("0"))
    LLIL_MUL = Column(Integer, server_default=text("0"))
    LLIL_MULS_DP = Column(Integer, server_default=text("0"))
    LLIL_MULU_DP = Column(Integer, server_default=text("0"))
    LLIL_NEG = Column(Integer, server_default=text("0"))
    LLIL_NOP = Column(Integer, server_default=text("0"))
    LLIL_NORET = Column(Integer, server_default=text("0"))
    LLIL_NOT = Column(Integer, server_default=text("0"))
    LLIL_OR = Column(Integer, server_default=text("0"))
    LLIL_POP = Column(Integer, server_default=text("0"))
    LLIL_PUSH = Column(Integer, server_default=text("0"))
    LLIL_REG = Column(Integer, server_default=text("0"))
    LLIL_REG_PHI = Column(Integer, server_default=text("0"))
    LLIL_REG_SPLIT = Column(Integer, server_default=text("0"))
    LLIL_REG_STACK_FREE_REG = Column(Integer, server_default=text("0"))
    LLIL_REG_STACK_FREE_REL = Column(Integer, server_default=text("0"))
    LLIL_REG_STACK_POP = Column(Integer, server_default=text("0"))
    LLIL_REG_STACK_PUSH = Column(Integer, server_default=text("0"))
    LLIL_REG_STACK_REL = Column(Integer, server_default=text("0"))
    LLIL_RET = Column(Integer, server_default=text("0"))
    LLIL_RLC = Column(Integer, server_default=text("0"))
    LLIL_ROL = Column(Integer, server_default=text("0"))
    LLIL_ROR = Column(Integer, server_default=text("0"))
    LLIL_ROUND_TO_INT = Column(Integer, server_default=text("0"))
    LLIL_RRC = Column(Integer, server_default=text("0"))
    LLIL_SBB = Column(Integer, server_default=text("0"))
    LLIL_SET_FLAG = Column(Integer, server_default=text("0"))
    LLIL_SET_REG = Column(Integer, server_default=text("0"))
    LLIL_SET_REG_SPLIT = Column(Integer, server_default=text("0"))
    LLIL_SET_REG_STACK_REL = Column(Integer, server_default=text("0"))
    LLIL_STORE = Column(Integer, server_default=text("0"))
    LLIL_SUB = Column(Integer, server_default=text("0"))
    LLIL_SX = Column(Integer, server_default=text("0"))
    LLIL_SYSCALL = Column(Integer, server_default=text("0"))
    LLIL_TAILCALL = Column(Integer, server_default=text("0"))
    LLIL_TEST_BIT = Column(Integer, server_default=text("0"))
    LLIL_TRAP = Column(Integer, server_default=text("0"))
    LLIL_UNDEF = Column(Integer, server_default=text("0"))
    LLIL_UNIMPL = Column(Integer, server_default=text("0"))
    LLIL_UNIMPL_MEM = Column(Integer, server_default=text("0"))
    LLIL_XOR = Column(Integer, server_default=text("0"))
    LLIL_ZX = Column(Integer, server_default=text("0"))


class MlilFeature(Base):
    __tablename__ = 'mlil_feature'

    function_id = Column(ForeignKey('functions.id'), primary_key=True)
    blocks = Column(Integer, server_default=text("0"))
    bytes = Column(Integer, server_default=text("0"))
    MLIL_NOP = Column(Integer, server_default=text("0"))
    MLIL_SET_VAR = Column(Integer, server_default=text("0"))
    MLIL_SET_VAR_FIELD = Column(Integer, server_default=text("0"))
    MLIL_SET_VAR_SPLIT = Column(Integer, server_default=text("0"))
    MLIL_LOAD = Column(Integer, server_default=text("0"))
    MLIL_LOAD_STRUCT = Column(Integer, server_default=text("0"))
    MLIL_STORE = Column(Integer, server_default=text("0"))
    MLIL_STORE_STRUCT = Column(Integer, server_default=text("0"))
    MLIL_VAR = Column(Integer, server_default=text("0"))
    MLIL_VAR_FIELD = Column(Integer, server_default=text("0"))
    MLIL_VAR_SPLIT = Column(Integer, server_default=text("0"))
    MLIL_ADDRESS_OF = Column(Integer, server_default=text("0"))
    MLIL_ADDRESS_OF_FIELD = Column(Integer, server_default=text("0"))
    MLIL_CONST = Column(Integer, server_default=text("0"))
    MLIL_CONST_PTR = Column(Integer, server_default=text("0"))
    MLIL_EXTERN_PTR = Column(Integer, server_default=text("0"))
    MLIL_FLOAT_CONST = Column(Integer, server_default=text("0"))
    MLIL_IMPORT = Column(Integer, server_default=text("0"))
    MLIL_ADD = Column(Integer, server_default=text("0"))
    MLIL_ADC = Column(Integer, server_default=text("0"))
    MLIL_SUB = Column(Integer, server_default=text("0"))
    MLIL_SBB = Column(Integer, server_default=text("0"))
    MLIL_AND = Column(Integer, server_default=text("0"))
    MLIL_OR = Column(Integer, server_default=text("0"))
    MLIL_XOR = Column(Integer, server_default=text("0"))
    MLIL_LSL = Column(Integer, server_default=text("0"))
    MLIL_LSR = Column(Integer, server_default=text("0"))
    MLIL_ASR = Column(Integer, server_default=text("0"))
    MLIL_ROL = Column(Integer, server_default=text("0"))
    MLIL_RLC = Column(Integer, server_default=text("0"))
    MLIL_ROR = Column(Integer, server_default=text("0"))
    MLIL_RRC = Column(Integer, server_default=text("0"))
    MLIL_MUL = Column(Integer, server_default=text("0"))
    MLIL_MULU_DP = Column(Integer, server_default=text("0"))
    MLIL_MULS_DP = Column(Integer, server_default=text("0"))
    MLIL_DIVU = Column(Integer, server_default=text("0"))
    MLIL_DIVU_DP = Column(Integer, server_default=text("0"))
    MLIL_DIVS = Column(Integer, server_default=text("0"))
    MLIL_DIVS_DP = Column(Integer, server_default=text("0"))
    MLIL_MODU = Column(Integer, server_default=text("0"))
    MLIL_MODU_DP = Column(Integer, server_default=text("0"))
    MLIL_MODS = Column(Integer, server_default=text("0"))
    MLIL_MODS_DP = Column(Integer, server_default=text("0"))
    MLIL_NEG = Column(Integer, server_default=text("0"))
    MLIL_NOT = Column(Integer, server_default=text("0"))
    MLIL_SX = Column(Integer, server_default=text("0"))
    MLIL_ZX = Column(Integer, server_default=text("0"))
    MLIL_LOW_PART = Column(Integer, server_default=text("0"))
    MLIL_JUMP = Column(Integer, server_default=text("0"))
    MLIL_JUMP_TO = Column(Integer, server_default=text("0"))
    MLIL_RET_HINT = Column(Integer, server_default=text("0"))
    MLIL_CALL = Column(Integer, server_default=text("0"))
    MLIL_CALL_UNTYPED = Column(Integer, server_default=text("0"))
    MLIL_CALL_OUTPUT = Column(Integer, server_default=text("0"))
    MLIL_CALL_PARAM = Column(Integer, server_default=text("0"))
    MLIL_RET = Column(Integer, server_default=text("0"))
    MLIL_NORET = Column(Integer, server_default=text("0"))
    MLIL_IF = Column(Integer, server_default=text("0"))
    MLIL_GOTO = Column(Integer, server_default=text("0"))
    MLIL_CMP_E = Column(Integer, server_default=text("0"))
    MLIL_CMP_NE = Column(Integer, server_default=text("0"))
    MLIL_CMP_SLT = Column(Integer, server_default=text("0"))
    MLIL_CMP_ULT = Column(Integer, server_default=text("0"))
    MLIL_CMP_SLE = Column(Integer, server_default=text("0"))
    MLIL_CMP_ULE = Column(Integer, server_default=text("0"))
    MLIL_CMP_SGE = Column(Integer, server_default=text("0"))
    MLIL_CMP_UGE = Column(Integer, server_default=text("0"))
    MLIL_CMP_SGT = Column(Integer, server_default=text("0"))
    MLIL_CMP_UGT = Column(Integer, server_default=text("0"))
    MLIL_TEST_BIT = Column(Integer, server_default=text("0"))
    MLIL_BOOL_TO_INT = Column(Integer, server_default=text("0"))
    MLIL_ADD_OVERFLOW = Column(Integer, server_default=text("0"))
    MLIL_SYSCALL = Column(Integer, server_default=text("0"))
    MLIL_SYSCALL_UNTYPED = Column(Integer, server_default=text("0"))
    MLIL_TAILCALL = Column(Integer, server_default=text("0"))
    MLIL_TAILCALL_UNTYPED = Column(Integer, server_default=text("0"))
    MLIL_BP = Column(Integer, server_default=text("0"))
    MLIL_TRAP = Column(Integer, server_default=text("0"))
    MLIL_INTRINSIC = Column(Integer, server_default=text("0"))
    MLIL_INTRINSIC_SSA = Column(Integer, server_default=text("0"))
    MLIL_FREE_VAR_SLOT = Column(Integer, server_default=text("0"))
    MLIL_FREE_VAR_SLOT_SSA = Column(Integer, server_default=text("0"))
    MLIL_UNDEF = Column(Integer, server_default=text("0"))
    MLIL_UNIMPL = Column(Integer, server_default=text("0"))
    MLIL_UNIMPL_MEM = Column(Integer, server_default=text("0"))
    MLIL_FADD = Column(Integer, server_default=text("0"))
    MLIL_FSUB = Column(Integer, server_default=text("0"))
    MLIL_FMUL = Column(Integer, server_default=text("0"))
    MLIL_FDIV = Column(Integer, server_default=text("0"))
    MLIL_FSQRT = Column(Integer, server_default=text("0"))
    MLIL_FNEG = Column(Integer, server_default=text("0"))
    MLIL_FABS = Column(Integer, server_default=text("0"))
    MLIL_FLOAT_TO_INT = Column(Integer, server_default=text("0"))
    MLIL_INT_TO_FLOAT = Column(Integer, server_default=text("0"))
    MLIL_FLOAT_CONV = Column(Integer, server_default=text("0"))
    MLIL_ROUND_TO_INT = Column(Integer, server_default=text("0"))
    MLIL_FLOOR = Column(Integer, server_default=text("0"))
    MLIL_CEIL = Column(Integer, server_default=text("0"))
    MLIL_FTRUNC = Column(Integer, server_default=text("0"))
    MLIL_FCMP_E = Column(Integer, server_default=text("0"))
    MLIL_FCMP_NE = Column(Integer, server_default=text("0"))
    MLIL_FCMP_LT = Column(Integer, server_default=text("0"))
    MLIL_FCMP_LE = Column(Integer, server_default=text("0"))
    MLIL_FCMP_GE = Column(Integer, server_default=text("0"))
    MLIL_FCMP_GT = Column(Integer, server_default=text("0"))
    MLIL_FCMP_O = Column(Integer, server_default=text("0"))
    MLIL_FCMP_UO = Column(Integer, server_default=text("0"))
    MLIL_SET_VAR_SSA = Column(Integer, server_default=text("0"))
    MLIL_SET_VAR_SSA_FIELD = Column(Integer, server_default=text("0"))
    MLIL_SET_VAR_SPLIT_SSA = Column(Integer, server_default=text("0"))
    MLIL_SET_VAR_ALIASED = Column(Integer, server_default=text("0"))
    MLIL_SET_VAR_ALIASED_FIELD = Column(Integer, server_default=text("0"))
    MLIL_VAR_SSA = Column(Integer, server_default=text("0"))
    MLIL_VAR_SSA_FIELD = Column(Integer, server_default=text("0"))
    MLIL_VAR_ALIASED = Column(Integer, server_default=text("0"))
    MLIL_VAR_ALIASED_FIELD = Column(Integer, server_default=text("0"))
    MLIL_VAR_SPLIT_SSA = Column(Integer, server_default=text("0"))
    MLIL_CALL_SSA = Column(Integer, server_default=text("0"))
    MLIL_CALL_UNTYPED_SSA = Column(Integer, server_default=text("0"))
    MLIL_SYSCALL_SSA = Column(Integer, server_default=text("0"))
    MLIL_SYSCALL_UNTYPED_SSA = Column(Integer, server_default=text("0"))
    MLIL_TAILCALL_SSA = Column(Integer, server_default=text("0"))
    MLIL_TAILCALL_UNTYPED_SSA = Column(Integer, server_default=text("0"))
    MLIL_CALL_OUTPUT_SSA = Column(Integer, server_default=text("0"))
    MLIL_CALL_PARAM_SSA = Column(Integer, server_default=text("0"))
    MLIL_LOAD_SSA = Column(Integer, server_default=text("0"))
    MLIL_LOAD_STRUCT_SSA = Column(Integer, server_default=text("0"))
    MLIL_STORE_SSA = Column(Integer, server_default=text("0"))
    MLIL_STORE_STRUCT_SSA = Column(Integer, server_default=text("0"))
    MLIL_VAR_PHI = Column(Integer, server_default=text("0"))
    MLIL_MEM_PHI = Column(Integer, server_default=text("0"))


class VexFeature(Base):
    __tablename__ = 'vex_feature'

    function_id = Column(ForeignKey('functions.id'), primary_key=True)
    blocks = Column(Integer, server_default=text("0"))
    bytes = Column(Integer, server_default=text("0"))
    iex_binop = Column(Integer, server_default=text("0"))
    iex_ccall = Column(Integer, server_default=text("0"))
    iex_const = Column(Integer, server_default=text("0"))
    iex_get = Column(Integer, server_default=text("0"))
    iex_geti = Column(Integer, server_default=text("0"))
    iex_gsptr = Column(Integer, server_default=text("0"))
    iex_ite = Column(Integer, server_default=text("0"))
    iex_load = Column(Integer, server_default=text("0"))
    iex_qop = Column(Integer, server_default=text("0"))
    iex_rdtmp = Column(Integer, server_default=text("0"))
    iex_triop = Column(Integer, server_default=text("0"))
    iex_unop = Column(Integer, server_default=text("0"))
    iex_vecret = Column(Integer, server_default=text("0"))
    ist_abihint = Column(Integer, server_default=text("0"))
    ist_cas = Column(Integer, server_default=text("0"))
    ist_dirty = Column(Integer, server_default=text("0"))
    ist_exit = Column(Integer, server_default=text("0"))
    ist_imark = Column(Integer, server_default=text("0"))
    ist_llsc = Column(Integer, server_default=text("0"))
    ist_loadg = Column(Integer, server_default=text("0"))
    ist_mbe = Column(Integer, server_default=text("0"))
    ist_noop = Column(Integer, server_default=text("0"))
    ist_put = Column(Integer, server_default=text("0"))
    ist_puti = Column(Integer, server_default=text("0"))
    ist_store = Column(Integer, server_default=text("0"))
    ist_storeg = Column(Integer, server_default=text("0"))
    ist_wrtmp = Column(Integer, server_default=text("0"))
    ico_u1 = Column(Integer, server_default=text("0"))
    ico_u8 = Column(Integer, server_default=text("0"))
    ico_u16 = Column(Integer, server_default=text("0"))
    ico_u32 = Column(Integer, server_default=text("0"))
    ico_u64 = Column(Integer, server_default=text("0"))
    ico_f32 = Column(Integer, server_default=text("0"))
    ico_f64 = Column(Integer, server_default=text("0"))
    ico_f64i = Column(Integer, server_default=text("0"))
    ico_v128 = Column(Integer, server_default=text("0"))
    ico_v256 = Column(Integer, server_default=text("0"))
    tmpvar = Column(Integer, server_default=text("0"))


class VexGraph(Base):
    __tablename__ = 'vex_graph'

    function_id = Column(ForeignKey('functions.id'), primary_key=True)
    ast = Column(LargeBinary)
    cdg = Column(LargeBinary)
    cfg = Column(LargeBinary)
    ddg = Column(LargeBinary)


class HlilBlock(Base):
    __tablename__ = 'hlil_block'

    function_id = Column(ForeignKey('functions.id'), primary_key=True)
    block_id = Column(Integer, primary_key=True, nullable=False)
    embedding = Column(Text)
    hlil = Column(Text)

    function = relationship('Function')


class VexBlock(Base):
    __tablename__ = 'vex_block'

    function_id = Column(ForeignKey('functions.id'), primary_key=True)
    block_id = Column(Integer, primary_key=True, nullable=False)
    embedding = Column(Text)

    function = relationship('Function')
