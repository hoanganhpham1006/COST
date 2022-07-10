import os
import errno
import numpy as np
import glob
import pickle

from torch.nn import init
import torch
import torch.nn as nn
from torch.autograd import Variable

def attention_func(attn_fuc, query, key, value, mask=None):
    feat = attn_fuc(query, key, value, mask) #Attention
    feat = torch.mean(feat, dim=1)
    return feat
    
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.sum()>0 and len(mask)>0: 
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, ae_generator, criterion, loss_l):
        self.generator = generator
        self.ae_generator = ae_generator
        self.criterion = criterion
        self.loss_l = loss_l
    
    def __call__(self, x, y, norm, ae_x=None, ae_y=None, ae_norm=None, query=None, encoded_query=None, query_mask=None, encoded_tgt=None):
        out = self.generator(x, query, encoded_query, query_mask, encoded_tgt)
        loss_1 = self.criterion(out.contiguous().view(-1, out.size(-1)), 
                              y.contiguous().view(-1))
        loss_2 = 0
        if ae_x is not None:
            for ae_x_i in ae_x:
                if self.ae_generator is not None:
                    ae_out = self.ae_generator(ae_x_i)
                else:
                    ae_out = self.generator(ae_x_i)
                loss_2 += self.criterion(ae_out.contiguous().view(-1, ae_out.size(-1)),
                                            ae_y.contiguous().view(-1))
        return loss_1 / norm.sum().float(), self.loss_l * loss_2 / ae_norm.sum().float()

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
    

def init_modules(modules, w_init='kaiming_uniform'):
    if w_init == "normal":
        _init = init.normal_
    elif w_init == "xavier_normal":
        _init = init.xavier_normal_
    elif w_init == "xavier_uniform":
        _init = init.xavier_uniform_
    elif w_init == "kaiming_normal":
        _init = init.kaiming_normal_
    elif w_init == "kaiming_uniform":
        _init = init.kaiming_uniform_
    elif w_init == "orthogonal":
        _init = init.orthogonal_
    else:
        raise NotImplementedError
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            _init(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        if isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.zeros_(param)
                elif 'weight' in name:
                    _init(param)

def save_model(model, optim, iter, model_dir, max_to_keep=None, model_name=""):
    checkpoint = {
        'iter': iter,
        'model': model.state_dict(),
        'optim': optim.state_dict() if optim is not None else None}
    if model_name == "":
        torch.save(checkpoint, "{}/checkpoint_{:06}.pth".format(model_dir, iter))
    else:
        torch.save(checkpoint, "{}/{}_checkpoint_{:06}.pth".format(model_dir, model_name, iter))

    if max_to_keep is not None and max_to_keep > 0:
        checkpoint_list = sorted([ckpt for ckpt in glob.glob(model_dir + "/" + '*.pth')])
        while len(checkpoint_list) > max_to_keep:
            os.remove(checkpoint_list[0])
            checkpoint_list = checkpoint_list[1:]


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def load_vocab(cfg):
    def invert_dict(d):
        return {v: k for k, v in d.items()}

    with open(os.path.join(cfg.DATASET.DATA_DIR, 'dic.pkl'), 'rb') as f:
        dictionaries = pickle.load(f)
    vocab = {}
    vocab['question_token_to_idx'] = dictionaries["word_dic"]
    # vocab['answer_token_to_idx'] = dictionaries["answer_dic"]
    vocab['question_token_to_idx']['pad'] = 0
    vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
    # vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])

    return vocab


def generateVarDpMask(shape, keepProb):
    randomTensor = torch.tensor(keepProb).cuda().expand(shape)
    randomTensor = randomTensor.clone() + nn.init.uniform_(torch.cuda.FloatTensor(shape[0], shape[1]))
    binaryTensor = torch.floor(randomTensor)
    mask = torch.cuda.FloatTensor(binaryTensor)
    return mask


def applyVarDpMask(inp, mask, keepProb):
    ret = (torch.div(inp, torch.tensor(keepProb).cuda())) * mask
    return ret