import torch

# from fairscale.optim.oss import OSS
from timm.optim import AdamP, RMSpropTF


def make_optimizer(cfg, model):
    """
    Create optimizer with per-layer learning rate and weight decay.
    """
    opt_name = cfg.OPT.OPTIMIZER
    eps = cfg.OPT.EPS

    params = []
    for key, value in model.named_parameters():
        lr = cfg.OPT.BASE_LR
        if not value.requires_grad:
            continue
        if "bertbase" in key:
            lr = cfg.OPT.BERT_LR
        weight_decay = cfg.OPT.WEIGHT_DECAY
        if "bias" in key:
            weight_decay = cfg.OPT.WEIGHT_DECAY_BIAS
        if "bn" in key or "layer_norm" in key or "norm" in key:
            weight_decay = cfg.OPT.WEIGHT_DECAY_BIAS
        if "gain" in key or "skipinit_gain" in key:
            weight_decay = cfg.OPT.WEIGHT_DECAY_BIAS
        if "pos_embed" in key or "cls_token" in key or "dist_token" in key:
            weight_decay = cfg.OPT.WEIGHT_DECAY_BIAS
        if "token_type_embed" in key or "text_embed" in key:
            weight_decay = cfg.OPT.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr, eps=eps)
    elif opt_name == "adamp":
        optimizer = AdamP(params, lr, eps=eps)
    elif opt_name == "rmsprop":
        optimizer = RMSpropTF(params, lr, alpha=0.9, momentum=0.9, eps=eps)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(params, lr, momentum=0.9)
    return optimizer


def make_oss(cfg, model):
    opt_name = cfg.OPT.OPTIMIZER
    eps = cfg.OPT.EPS

    base_arguments = {
        "lr": cfg.OPT.BASE_LR,
        "weight_decay": cfg.OPT.WEIGHT_DECAY,
    }

    if opt_name == "adamw":
        base_arguments.update({"eps": eps})
        base_optimizer = torch.optim.AdamW
    elif opt_name == "rmsprop":
        base_arguments.update({"alpha": 0.9, "momentum": 0.9, "eps": eps})
        base_optimizer = RMSpropTF
    elif opt_name == "sgd":
        base_arguments.update({"momentum": 0.9})
        base_optimizer = torch.optim.SGD

    optimizer = OSS(params=model.parameters(), optim=base_optimizer, **base_arguments)
    return optimizer
