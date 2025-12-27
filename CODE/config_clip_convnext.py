from yacs.config import CfgNode as CN
_C = CN()
# Base config files
_C.BASE = ['']
_C.SEED = 1
_C.init_lr = 3e-5
_C.batch_size = 128
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.f_mode = 'linear'
_C.MODEL.type = 'conv'
_C.MODEL.img_size = 224
_C.MODEL.num_classes = 4
_C.MODEL.finetune = None

_C.MODEL.output_dir = 'output/'
_C.MODEL.backbone = CN()
_C.MODEL.backbone.from_timm = True
_C.MODEL.backbone.model_name = 'convnext_base.clip_laion2b_augreg'
# _C.MODEL.backbone.model_name = 'convnext_base laion400m_s13b_b51k'
# _C.MODEL.backbone.model_name = 'convnext_base.fb_in22k'

_C.MODEL.backbone.out_dim = 1024
_C.MODEL.backbone.num_patch = 49
_C.MODEL.backbone.checkpoint = False
_C.MODEL.backbone.frozen = False

_C.MODEL.CLAdapter = CN()
_C.MODEL.CLAdapter.centers = 20
_C.MODEL.CLAdapter.layers = 1
_C.MODEL.CLAdapter.mlp_ratio = 4.0
_C.MODEL.CLAdapter.checkpoint = [False, 1]

#Optimizer
_C.Optimizer = CN()
_C.Optimizer.name = 'Adam'
_C.Optimizer.momentum = 0.9
_C.Optimizer.weight_decay = 1e-3

def get_config():
    config = _C.clone()
    return config