from collections import defaultdict, deque
import datetime
import pickle
import time
import json

import numpy as np
import torch
import torch.distributed as dist
from torch.autograd import Variable

import errno
import os

from pycocotools.coco import COCO
from .coco_eval import COCOEvalCap
from cvcore.modeling.utils import subsequent_mask


def greedy_decode(model, batch, device, max_len, start_symbol, pad_symbol):
    '''
        0video_idx, 1answers_in, 2answers_out, 3answers_len, 
        4questions, 5question_len, 
        6dialog, 7dialog_len, 
        8obj_app_feat, 9obj_spatial_feat, 
        10appearance_feat, 11motion_feat = batch
    '''

    encoder_out_list, encoder_pad_mask_list, _ = model.encode(batch[4].to(device), batch[6].to(device), \
                                                            batch[8].to(device), batch[9].to(device), \
                                                            batch[10].to(device), batch[11].to(device), \
                                                            batch[12].to(device), batch[13].to(device))
    bs = batch[4].shape[0]
    ys = torch.ones(bs, 1).fill_(start_symbol).type_as(batch[4].data).to(device)

    for i in range(max_len-1):
        output, _ = model.forward_features(encoder_out_list, encoder_pad_mask_list, \
                                                ys, \
                                                ae_use=None, ae_fts=[])
        prob = model.generator(output[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
    return ys

# def beam_search_decode(model, b, device, 
#                             max_len, 
#                             start_symbol, unk_symbol, end_symbol, pad_symbol, 
#                             beam=5, penalty=1.0, nbest=5, min_len=1):
#     '''
#         0video_idx, 1answers_in, 2answers_out, 3answers_len, 
#         4questions, 5question_len, 
#         6dialog, 7dialog_len, 
#         8obj_app_feat, 9obj_spatial_feat, 
#         10appearance_feat, 11motion_feat = batch
#     '''
#     query_mask = (b[4] != model.pad_id).unsqueeze(-2)
#      # permuted_ft = torch.from_numpy(b[11]).float().cuda().permute(1,0,2)
#     permuted_ft = b[11].to(device) 
#     fts_mask = [ (torch.sum(permuted_ft != 1, dim=2) != 0).unsqueeze(-2) ]
#     fts = [ (permuted_ft * fts_mask[0].squeeze().unsqueeze(-1).expand_as(b[11]).float()) ]
    
#     encoded_query, encoded_vid_features, encoded_cap, encoded_his, auto_encoded_ft = model.encode(b[4].to(device), query_mask.to(device), None, None, None, None, fts, fts_mask)


#     ds = torch.ones(1, 1).fill_(start_symbol).type_as(b[4].data)
#     hyplist=[([], 0., ds.to(device))]
#     best_state=None
#     comp_hyplist=[]

#     for l in range(max_len): 
#         new_hyplist = []
#         argmin = 0
#         for out, lp, st in hyplist:
#             output, _ = model.decode(encoded_vid_features, encoded_his, encoded_cap, encoded_query, fts_mask, None, None, query_mask.to(device), Variable(st), None, auto_encoded_ft)
#             if type(output) == tuple or type(output) == list:
#                 logp = model.generator(output[0][:, -1])
#             else:
#                 logp = model.generator(output[:, -1])
#             lp_vec = logp.cpu().data.numpy() + lp 
#             lp_vec = np.squeeze(lp_vec)
#             if l >= min_len:
#                 new_lp = lp_vec[end_symbol] + penalty * (len(out) + 1)
#                 comp_hyplist.append((out, new_lp))
#                 if best_state is None or best_state < new_lp: 
#                     best_state = new_lp
#             count = 1 
#             for o in np.argsort(lp_vec)[::-1]:
#                 if o == unk_symbol or o == end_symbol:
#                     continue 
#                 new_lp = lp_vec[o]
#                 if len(new_hyplist) == beam:
#                     if new_hyplist[argmin][1] < new_lp:
#                         new_st = torch.cat([st, torch.ones(1, 1).type_as(b[4].data).fill_(int(o)).to(device)], dim=1)
#                         new_hyplist[argmin] = (out + [o], new_lp, new_st)
#                         argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
#                     else:
#                         break
#                 else: 
#                     new_st = torch.cat([st, torch.ones(1, 1).type_as(b[4].data).fill_(int(o)).to(device)], dim=1)
#                     new_hyplist.append((out + [o], new_lp, new_st))
#                     if len(new_hyplist) == beam:
#                         argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
#                 count += 1
#         hyplist = new_hyplist 
            
#     if len(comp_hyplist) > 0:
#         maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
#         return maxhyps, best_state
#     else:
#         return [([], 0)], None

def beam_search_decode(model, batch, device, 
                            max_len, 
                            start_symbol, unk_symbol, end_symbol, pad_symbol, 
                            beam=5, penalty=1.0, nbest=5, min_len=1):
    '''
        0video_idx, 1answers_in, 2answers_out, 3answers_len, 
        4questions, 5question_len, 
        6dialog, 7dialog_len, 
        8obj_app_feat, 9obj_spatial_feat, 10obj_name_feat, 
        11appearance_feat, 12motion_feat = batch
    '''
    encoder_out_list, encoder_pad_mask_list, ae_fts, _, _ = model.encode(batch[4].to(device), batch[6].to(device), batch[1].to(device),\
                                                                            batch[8].to(device), batch[9].to(device), \
                                                                            batch[10].to(device), batch[11].to(device), \
                                                                            batch[12].to(device), batch[13].to(device))

    ds = torch.ones(1, 1).fill_(start_symbol).type_as(batch[4].data)
    hyplist=[([], 0., ds.to(device))]
    best_state=None
    comp_hyplist=[]

    for l in range(max_len): 
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            # output, _ = model.forward_features(encoder_out_list, encoder_pad_mask_list, \
            #                                     Variable(st), \
            #                                     ae_use=model.ae_use, ae_fts=ae_fts)
            tgt_attn_mask = (Variable(st) != pad_symbol).unsqueeze(-2)
            tgt_attn_mask = tgt_attn_mask & Variable(
                subsequent_mask(Variable(st).size(-1)).type_as(tgt_attn_mask.data))
            tgt_embeds = model.tgt_embed(Variable(st))
            output, _ = model.forward_features(encoder_out_list, encoder_pad_mask_list, \
                                                tgt_embeds, tgt_attn_mask,\
                                                ae_use=model.ae_use, ae_fts=ae_fts)
            if type(output) == tuple or type(output) == list:
                logp = model.generator(output[0][:, -1:], batch[4].to(device), encoder_out_list[1], encoder_pad_mask_list[1], tgt_embeds[:, -1:])
            else:
                logp = model.generator(output[:, -1:], batch[4].to(device), encoder_out_list[1], encoder_pad_mask_list[1], tgt_embeds[:, -1:])
            lp_vec = logp.cpu().data.numpy() + lp 
            lp_vec = np.squeeze(lp_vec)
            if l >= min_len:
                new_lp = lp_vec[end_symbol] + penalty * (len(out) + 1)
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state < new_lp: 
                    best_state = new_lp
            count = 1 
            for o in np.argsort(lp_vec)[::-1]:
                if o == unk_symbol or o == end_symbol:
                    continue 
                new_lp = lp_vec[o]
                if len(new_hyplist) == beam:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = torch.cat([st, torch.ones(1, 1).type_as(batch[4].data).fill_(int(o)).to(device)], dim=1)
                        new_hyplist[argmin] = (out + [o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
                    else:
                        break
                else: 
                    new_st = torch.cat([st, torch.ones(1, 1).type_as(batch[4].data).fill_(int(o)).to(device)], dim=1)
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == beam:
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
                count += 1
        hyplist = new_hyplist 
            
    if len(comp_hyplist) > 0:
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
        return maxhyps, best_state
    else:
        return [([], 0)], None

def cocoval(test, out_path, preds, gts, gts_path=None, map_id_name=None):
    #Hyp save
    hyp_file = out_path + "_tmp_hyp.json"
    annos = []
    existed = []
    # image_id=1
    for results in preds:
        for pred in results: #Multi gpu
            if pred[0] in existed:
                continue
            annos.append({'image_id': pred[0], 'caption': pred[1]})
            existed.append(pred[0])
        # image_id += 1
    json.dump(annos, open(hyp_file,'w'), indent=4)

    #Ref save
    if gts is not None:
        ref_file = "_tmp_ref.json"
        data = {}
        data['info'] = {}
        data['licenses'] = []
        annos = []
        images = []
        for results in gts:
            for cap_id, gt in results: #Multi gpu
                annos.append({"image_id": cap_id, "id": cap_id, "caption": gt})
                images.append({"name": str(cap_id).zfill(5) + "_1", "id": cap_id})
        data['images'] = images
        data['type'] = 'captions'
        data['annotations'] = annos
        json.dump(data, open(ref_file, 'w'), indent=4)
    
    elif gts_path is not None and map_id_name is not None:
        ref_file = gts_path
        data_test = json.load(open(gts_path))
        map_name_org_id = {}
        for i in range(len(data_test['images'])):
            map_name_org_id[data_test['images'][i]['name'].split('_')[0]] = data_test['images'][i]['id']
        for i in range(len(annos)):
            origin_name = map_id_name[annos[i]['image_id']]
            annos[i]['image_id'] = map_name_org_id[origin_name]
        json.dump(annos, open(hyp_file,'w'), indent=4)
    
    # Get results
    coco = COCO(ref_file)
    cocoRes = coco.loadRes(hyp_file)
    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes)
    # evaluate on a subset of images by setting
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    # evaluate results
    cocoEval.evaluate(test)
    # print output evaluation scores
    for metric, score in cocoEval.eval.items():
        print('%s: %.3f'%(metric, score))
    return cocoEval.eval['Bleu_4']

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def collate_fn(batch):
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
