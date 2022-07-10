r"""Classification Training.
"""
import datetime
import os
import time
import sys

import torch
from torch.cuda.amp import autocast, GradScaler
import torch.utils.data
from torch import nn

from cvcore.config import get_cfg
from cvcore.modeling.COST import COST
from cvcore.solver import make_lr_scheduler, make_optimizer
from timm.data.loader import PrefetchLoader
from timm.utils.model_ema import ModelEmaV2
from tools.engine import (
    train_one_epoch,
    evaluate_classifier,
    inferencing,
    freeze_model
)
import tools.utils as utils
from cvcore.datasets.AVSD import AVSDDataset
from cvcore.utils.batching import collate_batch

class MultiPrefetchLoader(PrefetchLoader):
    def __iter__(self):
        stream = torch.cuda.Stream()
        for data in self.loader:
            with torch.cuda.stream(stream):
                for d in data:
                    d = d.cuda(non_blocking=True)
                    if self.fp16:
                        d = d.half()
                    else:
                        d = d.float()
                    if self.random_erasing is not None:
                        d = self.random_erasing(d)
            torch.cuda.current_stream().wait_stream(stream)
            yield data

def create_args():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--print-freq", default=50, type=int, help="print frequency")
    parser.add_argument("--aspect-ratio-group-factor", default=-1, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--finetune",
        "-ft",
        action="store_true",
        help="whether to attempt to resume from the checkpoint optimizer, scheduler and epoch",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--bucket",
        dest="bucket",
        help="Use bucket sampler for faster training",
        action="store_true",
    )
    parser.add_argument(
        "--clip-grad",
        dest="clip_grad",
        help="apply gradient clipping",
        action="store_true",
    )
    parser.add_argument(
        "--beam-size", default=3, type=int, help="number of beam hypotheses"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # Language model inference time
    parser.add_argument(
        "--num-beams", default=3, help="number of beam hypotheses", type=int
    )
    parser.add_argument(
        "--num-returns", default=1, help="number of sequences returned", type=int
    )

    # distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )

    args = parser.parse_args()
    return args


def main(args, cfg):
    utils.init_distributed_mode(args)
    print(args)
    imgs_per_gpu = cfg.TRAIN.BATCH_SIZE // args.world_size
    workers_per_gpu = cfg.SYSTEM.NUM_WORKERS // args.world_size

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")
    dataset = AVSDDataset(cfg.DIRS, "train")
    dataset_test = AVSDDataset(cfg.DIRS, "val")
        
    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test, shuffle=False
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, imgs_per_gpu, drop_last=False
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=train_batch_sampler,
        num_workers=workers_per_gpu,
        collate_fn=collate_batch,
    )
    data_loader = MultiPrefetchLoader(
        data_loader,
        fp16=False,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TRAIN.BATCH_SIZE//2,
        sampler=test_sampler,
        num_workers=workers_per_gpu,
        collate_fn=collate_batch,
    )

    data_loader_test = MultiPrefetchLoader(
        data_loader_test,
        fp16=False,
    )

    print("Creating model")
    model = COST(cfg.MODEL, dataset.vocab)
    # model = make_model(cfg.MODEL, dataset.vocab) 
    # freeze_model(cfg, model)
    print(
        f"Created model {cfg.EXPERIMENT} - param count:{sum([m.numel() for m in model.parameters()])}"
    )
    model.to(device)

    scaler = GradScaler()
    optimizer = make_optimizer(cfg, model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    lr_scheduler = make_lr_scheduler(cfg, optimizer, data_loader)

    best_metric = 10000

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if args.finetune:
            print("Skip loading optimizer and scheduler state dicts")
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            best_metric = checkpoint["best_metric"]
            print(f"Epoch: {args.start_epoch} - Best metric: {best_metric}")

    # Checkpoint name
    save_filename = cfg.EXPERIMENT
    save_filename += ".pth"

    if not args.test_only:
        print("Start training")
        start_time = time.time()

        for epoch in range(args.start_epoch, cfg.TRAIN.EPOCHS):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            train_one_epoch(
                model,
                optimizer,
                lr_scheduler,
                data_loader,
                device,
                epoch,
                args.print_freq,
                gd_steps=cfg.OPT.GD_STEPS,
                scaler=scaler,
                clip_grad=args.clip_grad,
            )
            # evaluate_classifier after every epoch
            epoch_metric = evaluate_classifier(
                model, data_loader_test, device=device
            )
            # preds, gts = evaluate_classifier(model_without_ddp, data_loader_test, device)
            # val_bleu_4 = utils.cocoval(False, cfg.DIRS.OUTPUTS + cfg.EXPERIMENT, preds, gts=gts, \
            #                 gts_path=None, map_id_name=dataset_test.map_id_name)
            checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "args": args,
                    "epoch": epoch,
                    "best_metric": epoch_metric,
                }
            utils.save_on_master(
                checkpoint,
                os.path.join(cfg.DIRS.WEIGHTS, f"e{epoch}_{save_filename}"),
            )

            if epoch_metric < best_metric:
                best_metric = epoch_metric
                print(f"Saving to {os.path.join(cfg.DIRS.WEIGHTS, 'best_' + save_filename)}")
                utils.save_on_master(
                    checkpoint,
                    os.path.join(cfg.DIRS.WEIGHTS, f"best_{save_filename}"),
                )
            

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    print("Start testing")
    del dataset_test, dataset
    start_time = time.time()
    dataset_test = AVSDDataset(cfg.DIRS, "test")
    print(f"Creating datatest loaders @ {args.world_size} GPUs")
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test, shuffle=False
        )
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=test_sampler,
        num_workers=workers_per_gpu,
        collate_fn=collate_batch,
    )

    if args.resume == "":
        checkpoint = torch.load(f"{cfg.DIRS.WEIGHTS}best_{save_filename}", map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        print(f"Loaded epoch: {checkpoint['epoch'] + 1} - Best metric: {checkpoint['best_metric']}")

    preds = inferencing(model_without_ddp, data_loader_test, device)
    print(f"Number test sample: {sum([len(pred) for pred in preds])}")
    _ = utils.cocoval(True, cfg.DIRS.OUTPUTS + cfg.EXPERIMENT, preds, gts=None, \
                    gts_path=cfg.DIRS.INPUT_DIR + cfg.DIRS.TEST_REF_JSON, map_id_name=dataset_test.map_id_name)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Testing time {}".format(total_time_str))

    sys.exit(0)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    args = create_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    # Make dirs
    for _dir in ["WEIGHTS", "OUTPUTS", "LOGS"]:
        if not os.path.isdir(cfg.DIRS[_dir]):
            os.makedirs(cfg.DIRS[_dir], exist_ok=True)
    main(args, cfg)
