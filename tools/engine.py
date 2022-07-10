import math
import sys
import time

from tqdm import tqdm

import torch
from torch.cuda.amp import autocast
import numpy as np

from timm.utils import dispatch_clip_grad

from . import utils


def freeze_model(cfg, model):
    """
    Freeze some or all parts of the model.
    """
    # frozen_layers = cfg.MODEL.FREEZE_AT
    frozen_layers = ["bertbase"]

    if len(frozen_layers) > 0:
        for name, parameter in model.named_parameters():
            if any([name.startswith(layer) for layer in frozen_layers]):
                print("Freezing layer: {}".format(name))
                parameter.requires_grad_(False)


def train_one_epoch(
    model,
    optimizer,
    scheduler,
    data_loader,
    device,
    epoch,
    print_freq,
    gd_steps=1,
    scaler=None,
    clip_grad=False,
    model_ema=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    for i, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        with autocast(False):
            loss_dict = model(batch)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # gradient accumulation
        losses = losses / gd_steps
        scaler.scale(losses).backward()
        if (i + 1) % gd_steps == 0:
            if clip_grad:
                scaler.unscale_(optimizer)
                dispatch_clip_grad(model.parameters(), value=1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return metric_logger


@torch.no_grad()
def evaluate_classifier(model, data_loader, device):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Evaluate:"

    losses_dict = {}

    for i, batch in enumerate(metric_logger.log_every(data_loader, 100, header)):
        torch.cuda.synchronize()
        model_time = time.time()
        with autocast(False):
            loss_seq = model(batch)
        model_time = time.time() - model_time

        evaluator_time = time.time()
        losses_dict.update(loss_seq)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    seq_loss = losses_dict['anwser loss'].cpu().data.numpy()
    seq_loss = np.mean(utils.all_gather(seq_loss))

    print(f"Valid loss: {seq_loss} \n ")
    return seq_loss

# @torch.no_grad()
# def evaluate_classifier(model, data_loader, device):
#     vocablist = sorted(model.vocab.keys(), key=lambda s:model.vocab[s])
#     model.eval()
#     header = "Evaluate:"
#     preds = []
#     gts = []
#     print_freq = 5
#     count = 1
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     evaluator_time = time.time()
#     for ind, batch in enumerate(
#         metric_logger.log_every(data_loader, print_freq, header)
#     ):  
#         question = batch[4]
#         answer_out = batch[2]
#         pred_outs = utils.greedy_decode(model, batch, device, max_len=30, \
#                                                             start_symbol=model.start_id, \
#                                                             pad_symbol = model.pad_id)
#         device_index = pred_outs.device.index
#         if ind % print_freq == 0:
#             quesstr = []
#             for w in question[0][1:-1]:
#                 if w == model.end_id:
#                     break
#                 quesstr.append(vocablist[w])
            
#             quesstr = " ".join(quesstr)
#             print('QUS: ' + quesstr)
        
#         for i in range(len(pred_outs)):
#             ansstr = []
#             for w in answer_out[i]:
#                 if w == model.end_id:
#                     break
#                 ansstr.append(vocablist[w])
#             ansstr = " ".join(ansstr)

#             hyp = []
#             for n in range(len(pred_outs[i])):
#                 w = pred_outs[i][n].item()
#                 if w == model.end_id or w == model.unk_id:
#                     break
#                 hyp.append(vocablist[w])
#             hyp = " ".join(hyp[1:])
#             if ind % print_freq == 0 and i == 0:
#                 print('HYP: %s' % (hyp))
#                 print('REF: ' + ansstr)
#             preds.append((count + device_index*10000, hyp))  
#             gts.append((count + device_index*10000, ansstr))
#             count += 1
#         evaluator_time = time.time() - evaluator_time
#         metric_logger.update(evaluator_time=evaluator_time)
#     return utils.all_gather(preds), utils.all_gather(gts)


@torch.no_grad()
def inferencing(model, data_loader, device):
    vocablist = sorted(model.vocab.keys(), key=lambda s:model.vocab[s])
    model.eval()
    header = "Inference:"
    preds = []
    # gts = []
    nbest = 3
    print_freq = 200
    metric_logger = utils.MetricLogger(delimiter="  ")

    for i, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):  
        video_idx = batch[0]
        question = batch[4]
        pred_out, _ = utils.beam_search_decode(model, batch, device,
                                                max_len=30, 
                                                start_symbol=model.start_id, 
                                                unk_symbol=model.unk_id, 
                                                end_symbol=model.end_id, 
                                                pad_symbol=model.pad_id,
                                                beam=3)
        if i % print_freq == 0:
            quesstr = []
            for w in question[0][1:-1]:
                if w == model.end_id:
                    break
                quesstr.append(vocablist[w])
            
            quesstr = " ".join(quesstr)
            print('QUS: ' + quesstr)

        besthyp = ''
        for n in range(min(nbest, len(pred_out))):
            pred = pred_out[n]
            hypstr = []
            for w in pred[0]:
                if w == model.end_id or w == model.unk_id:
                    break
                hypstr.append(vocablist[w])
            hypstr = " ".join(hypstr)
            if n == 0:
                besthyp = hypstr
            if i % print_freq == 0:
                print('HYP[%d]: %s  ( %f )' % (n + 1, hypstr, pred[1]))
        preds.append((video_idx[0].item(), besthyp))
    return utils.all_gather(preds)
