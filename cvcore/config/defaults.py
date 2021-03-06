from yacs.config import CfgNode as CN


_C = CN()

# Experiment name
_C.EXPERIMENT = ""
# Sub-sample training dataset to debug
_C.DEBUG = False

_C.SYSTEM = CN()
_C.SYSTEM.SEED = 0
_C.SYSTEM.CUDA = True
_C.SYSTEM.DISTRIBUTED = False
_C.SYSTEM.MULTI_GPU = False
_C.SYSTEM.NUM_WORKERS = 8

_C.DIRS = CN()
_C.DIRS.INPUT_DIR = "./data/"
_C.DIRS.TRAIN_QUESTION_PT = ""
_C.DIRS.VAL_QUESTION_PT = ""
_C.DIRS.TEST_QUESTION_PT = ""
_C.DIRS.TEST_REF_JSON = ""
_C.DIRS.VOCAB_FILE = ""
_C.DIRS.OBJECT_FEATURE_H5 = ""
_C.DIRS.RESNET_FEATURE_H5 = ""
_C.DIRS.I3D_FEATURE_H5 = ""
_C.DIRS.MAP_ID_NAME = ""
_C.DIRS.WEIGHTS = "weights/"
_C.DIRS.LOGS = "logs/"
_C.DIRS.OUTPUTS = "outputs/"

_C.MODEL = CN()
_C.MODEL.NUM_BLOCKS = 1
_C.MODEL.NUM_HEADS = 1
_C.MODEL.D_FF = 512
_C.MODEL.D_MODEL = 512
_C.MODEL.DROPOUT = 0.1
_C.MODEL.APP_FEAT = 512
_C.MODEL.MOTION_FEAT = 512
_C.MODEL.OBJ_APP_FEAT = 512
_C.MODEL.TRAIN_WEIGHT_INIT = "xavier_uniform"
_C.MODEL.TRAIN_VAR_DROPOUT = False
_C.MODEL.TRAIN_CLIP_GRADS = True
_C.MODEL.AE_USE = False
_C.MODEL.FREEZE_AT = []

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 1
_C.TRAIN.BATCH_SIZE = 1

_C.SYSTEM = CN()
_C.SYSTEM.NUM_WORKERS = 8

_C.OPT = CN()
_C.OPT.OPTIMIZER = "adamw"
_C.OPT.GD_STEPS = 1
_C.OPT.WARMUP_EPOCHS = 1
_C.OPT.BASE_LR = 1e-3
_C.OPT.BACKBONE_LR = 1e-3
_C.OPT.BERT_LR = 1e-3
_C.OPT.WEIGHT_DECAY = 1e-3
_C.OPT.WEIGHT_DECAY_BIAS = 0.0
_C.OPT.EPS = 1e-4
# RMSpropTF options
_C.OPT.DECAY_EPOCHS = 2.4
_C.OPT.DECAY_RATE = 0.97
# StepLR scheduler
_C.OPT.MILESTONES = [10, 20]
# Learning rate scheduler
_C.OPT.SCHED = "cosine_warmup"

_C.OPT.SWA = CN({"ENABLED": False})
_C.OPT.SWA.DECAY_RATE = 0.999
_C.OPT.SWA.START = 10
_C.OPT.SWA.FREQ = 5

