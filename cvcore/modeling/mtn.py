import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math, copy, time
from torch.autograd import Variable

class EncoderDecoder(nn.Module):
    def __init__(self, query_encoder, his_encoder, cap_encoder, vid_encoder, decoder, query_embed, his_embed, cap_embed, tgt_embed, generator, diff_encoder=False, auto_encoder_embed=None, auto_encoder_ft=None, auto_encoder_generator=None):
        super(EncoderDecoder, self).__init__() 
        self.query_encoder = query_encoder
        self.his_encoder = his_encoder
        self.cap_encoder = cap_encoder
        self.vid_encoder = vid_encoder
        self.decoder = decoder
        self.query_embed = query_embed
        self.his_embed = his_embed
        self.cap_embed = cap_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.diff_encoder = diff_encoder
        self.auto_encoder_embed = auto_encoder_embed
        self.auto_encoder_ft=auto_encoder_ft
        self.auto_encoder_generator=auto_encoder_generator

    def forward(self, b):
        encoded_query, encoded_vid_features, encoded_cap, encoded_his, auto_encoded_ft = self.encode(b.query, b.query_mask, b.his, b.his_mask, b.cap, b.cap_mask, b.fts, b.fts_mask)
        output = self.decode(encoded_vid_features, encoded_his, encoded_cap, encoded_query, b.fts_mask, b.his_mask, b.cap_mask, b.query_mask, b.trg, b.trg_mask, auto_encoded_ft)
        return output

    def vid_encode(self, video_features, video_features_mask, encoded_query=None):
        output = []
        for i, ft in enumerate(video_features):
            output.append(self.vid_encoder[i](ft))
        return output

    def encode(self, query, query_mask, his=None, his_mask=None, cap=None, cap_mask=None, vid=None, vid_mask=None):
        if self.diff_encoder:
            if self.auto_encoder_ft == 'caption' or self.auto_encoder_ft == 'summary':
                ft = cap
            elif self.auto_encoder_ft == 'query':
                ft = query
            if self.auto_encoder_embed is not None:
                ae_encoded = []
                for i in range(len(vid)):
                    ae_encoded.append(self.auto_encoder_embed[i](ft))
            else:
                ae_encoded = []
                for i in range(len(vid)):
                    ae_encoded.append(self.query_embed(ft))
            return self.query_encoder(self.query_embed(query), self.vid_encode(vid, vid_mask), self.query_embed(cap), self.query_embed(his), ae_encoded)
        else:
            output = self.query_encoder(self.query_embed(query), self.vid_encode(vid, vid_mask), self.query_embed(cap), self.query_embed(his))
            output.append(None)
            return output 

    def decode(self, encoded_vid_features, his_memory, cap_memory, query_memory, vid_features_mask, his_mask, cap_mask, query_mask, tgt, tgt_mask, auto_encoded_ft):
        # tgt 32, 19
        encoded_tgt = self.tgt_embed(tgt) # 32, 19, 512
        return self.decoder(encoded_vid_features, vid_features_mask, encoded_tgt, his_memory, his_mask, cap_memory, cap_mask, query_memory, query_mask, tgt_mask, auto_encoded_ft, self.auto_encoder_ft)

# class Generator(nn.Module):
#     "Define standard linear + softmax generation step."
#     def __init__(self, d_model, vocab):
#         super(Generator, self).__init__()
#         self.proj = nn.Linear(d_model, vocab)

#     def forward(self, x):
#         import pdb;pdb.set_trace()
#         return F.log_softmax(self.proj(x), dim=-1)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab, W=None):
        super(Generator, self).__init__()
        if W is not None:
            self.proj = W
        else:
            self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # return F.log_softmax(self.proj(x), dim=-1)
        out = x.matmul(self.proj.transpose(1,0))
        return F.log_softmax(out, dim=-1)

class PointerGenerator(nn.Module):
    def __init__(self, d_model, vocab_gen, pointer_attn):
        super(PointerGenerator, self).__init__()
        self.vocab_gen = vocab_gen
        self.pointer_gen_W = nn.Linear(d_model*3, 1) 
        self.pointer_attn = pointer_attn 

    def forward(self, x, query, encoded_query, query_mask, encoded_tgt):
        logits = x
        vocab_attn =  logits.matmul(self.vocab_gen.transpose(1,0))
        p_vocab = F.softmax(vocab_attn, dim = -1)

        text = query
        encoded_text = encoded_query
        text_mask = query_mask
        encoded_in = encoded_tgt
        
        self.pointer_attn(logits, encoded_text, encoded_text, text_mask)
        pointer_attn = self.pointer_attn.attn.squeeze(1)
        
        text_index = text.unsqueeze(1).expand_as(pointer_attn)
        p_text_ptr = torch.zeros(p_vocab.size()).cuda()
        p_text_ptr.scatter_add_(2, text_index, pointer_attn)
                        
        expanded_pointer_attn = pointer_attn.unsqueeze(-1).repeat(1, 1, 1, encoded_text.shape[-1])
        text_vec = (encoded_text.unsqueeze(1).expand_as(expanded_pointer_attn) * expanded_pointer_attn).sum(2)
        p_gen_vec = torch.cat([logits, text_vec, encoded_in], -1)
        vocab_pointer_switches = nn.Sigmoid()(self.pointer_gen_W(p_gen_vec)).expand_as(p_text_ptr)
        p_out = (1 - vocab_pointer_switches) * p_text_ptr + vocab_pointer_switches * p_vocab
        return torch.log(p_out+1e-30)
        

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size"
        return x + self.dropout(sublayer(self.norm(x)))

    def expand_forward(self, x, sublayer):
        out = self.dropout(sublayer(self.norm(x)))
        out = out.mean(1).unsqueeze(1).expand_as(x)
        return x + out 

    def nosum_forward(self, x, sublayer):
        return self.dropout(sublayer(self.norm(x)))

class Encoder(nn.Module):
    def __init__(self, sizes):
        super(Encoder, self).__init__()
        self.norm = nn.ModuleList()
        for size in sizes:
            self.norm.append(LayerNorm(size))

    def forward(self, layers):
        output = []
        i = 0
        for layer in layers:
            output.append(self.norm[i](layer))
            # output.append(layer)
            i += 1
        return output

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, ff1, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.ff1 = ff1
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size 

    def forward(self, seq, seq_mask):
        seq = self.sublayer[0](seq, lambda seq: self.self_attn(seq, seq, seq, seq_mask))
        return self.sublayer[1](seq, self.ff1)

class Decoder(nn.Module):
    def __init__(self, layer, N, ft_sizes):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

        self.ae_norm = nn.ModuleList()
        for ft_size in ft_sizes:
            self.ae_norm.append(LayerNorm(ft_size))

    def forward(self, x, tgt_mask, memory_list, pad_mask_list, ae_use, ae_ft):
        for layer in self.layers:
            x, ae_ft = layer(x, tgt_mask, memory_list, pad_mask_list, ae_use, ae_ft)
        out_ae_ft = []
        for i, ft in enumerate(ae_ft):
            out_ae_ft.append(self.ae_norm[i](ft))
        return self.norm(x), out_ae_ft
        # return x, out_ae_ft

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, attn_ques, feed_forward, 
                # attn_dialog, attn_history,
                attn_dialog,
                # ae_motion_self_attn, ae_motion_attn, ae_motion_feed_forward, ae_motion_attn_ques,
                ae_app_self_attn, ae_app_attn, ae_app_feed_forward, ae_app_attn_ques, 
                dropout):
                # combine_att, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size

        self.self_attn = self_attn
        self.src_attn_ques = attn_ques
        self.src_attn_dialog = attn_dialog
        self.feed_forward = feed_forward

        
        self.auto_encoder_app_attn = ae_app_attn_ques
        self.auto_encoder_app_feed_forward = ae_app_feed_forward
        self.src_attn_app = ae_app_attn
        self.auto_encoder_app_self_attn = ae_app_self_attn

        # self.src_attn_his = attn_history
        # self.auto_encoder_motion_self_attn = ae_motion_self_attn
        # self.auto_encoder_motion_attn = ae_motion_attn_ques
        # self.auto_encoder_motion_feed_forward = ae_motion_feed_forward
        # self.src_motion_app = ae_motion_attn

        # self.combine_att = combine_att
        # self.vc_combine_W = nn.Linear(size*3, 1)

        self.sublayer = clones(SublayerConnection(size, dropout), 4 + 4)

    def forward(self, x, tgt_mask, memory_list, memory_pad_mask_list, ae_use, ae_ft):
        assert len(memory_list) == len(memory_pad_mask_list)
        count = 0
        x = self.sublayer[count](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        count += 1

        i = 0
        x = self.sublayer[count](x, lambda x: self.src_attn_dialog(x, memory_list[i], memory_list[i], memory_pad_mask_list[i]))
        count += 1; i += 1

        x = self.sublayer[count](x, lambda x: self.src_attn_ques(x, memory_list[i], memory_list[i], memory_pad_mask_list[i]))
        count += 1; i += 1  

        # x = self.sublayer[count](x, lambda x: self.src_attn_his(x, memory_list[i], memory_list[i], memory_pad_mask_list[i]))
        # count += 1; i += 1        
        
        ae_out = []
         
        #### 

        if ae_use:
            ae_x_1 = ae_ft[0]
            ae_tgt_mask = memory_pad_mask_list[1]
            ae_x_1 = self.sublayer[count](ae_x_1, lambda ae_x_1: self.auto_encoder_app_self_attn(ae_x_1, ae_x_1, ae_x_1, ae_tgt_mask))
            count += 1
            ae_x_1 = self.sublayer[count](ae_x_1, lambda ae_x_1: self.auto_encoder_app_attn(ae_x_1, memory_list[i], memory_list[i], memory_pad_mask_list[i]))
            count += 1; i += 1
            ae_x_1 = self.sublayer[count](ae_x_1, self.auto_encoder_app_feed_forward)
            count += 1
            ae_out.append(ae_x_1)
            x = self.sublayer[count](x, lambda x: self.src_attn_app(x, ae_x_1, ae_x_1, ae_tgt_mask))
            count += 1
            

            # ae_x_2 = ae_ft[1]
            # ae_x_2 = self.sublayer[count](ae_x_2, lambda ae_x_2: self.auto_encoder_motion_self_attn(ae_x_2, ae_x_2, ae_x_2, ae_tgt_mask))
            # count += 1
            # ae_x_2 = self.sublayer[count](ae_x_2, lambda ae_x_2: self.auto_encoder_motion_attn(ae_x_2, memory_list[i], memory_list[i], memory_pad_mask_list[i]))
            # count += 1; i += 1
            # ae_x_2 = self.sublayer[count](ae_x_2, self.auto_encoder_motion_feed_forward)
            # count += 1
            # ae_out.append(ae_x_2)
            # ae_x_2 = self.sublayer[count](x, lambda x: self.src_motion_app(x, ae_x_2, ae_x_2, ae_tgt_mask))
            # count += 1

            # Combined
            # temp = torch.cat([x, ae_x_1, ae_x_2], dim=-1)
            # combine_score = nn.Sigmoid()(self.vc_combine_W(temp))
            # x = combine_score*ae_x_1 + (1-combine_score)*ae_x_2
            
        else:
            x = self.sublayer[count](x, lambda x: self.auto_encoder_app_attn(x, memory_list[i], memory_list[i], memory_pad_mask_list[i]))
            count += 1; i += 1

        x = self.sublayer[count](x, self.feed_forward)
        return x, ae_out

class DecoderSimple(nn.Module):
    def __init__(self, layer, N, ft_sizes):
        super(DecoderSimple, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

        # self.ae_norm = nn.ModuleList()
        # for ft_size in ft_sizes:
        #     self.ae_norm.append(LayerNorm(ft_size))

    def forward(self, x, tgt_mask, memory, pad_mask):
        for layer in self.layers:
            x = layer(x, tgt_mask, memory, pad_mask)
        return self.norm(x)[:, -1:]
        # return x[:, -1:]

class DecoderLayerSimple(nn.Module):
    def __init__(self, size, self_attn, attn_ques, feed_forward, dropout):
        super(DecoderLayerSimple, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn_ques = attn_ques
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, tgt_mask, memory, memory_pad_mask):
        count = 0
        x = self.sublayer[count](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        count += 1

        x = self.sublayer[count](x, lambda x: self.src_attn_ques(x, memory, memory, memory_pad_mask))
        count += 1     
        
        x = self.sublayer[count](x, self.feed_forward)
        return x


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, d_in=-1, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        if d_in < 0: 
            d_in = d_model 
        self.linears = clones(nn.Linear(d_in, d_model), 3)
        self.linears.append(nn.Linear(d_model, d_in))
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1, d_out=-1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        if d_out < 0:
            d_out = d_model 
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class StPositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=50):
        super(StPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
            
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
  
    def forward(self, x, x_st):
        x = x + Variable(self.pe[:, x_st], requires_grad=False)
        x = x.squeeze(0)
        return self.dropout(x)

def make_model(src_vocab, tgt_vocab, 
    N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, 
    separate_his_embed=False, separate_cap_embed=False, 
    ft_sizes=None, 
    diff_encoder=False, diff_embed=False, diff_gen=False, 
    auto_encoder_ft=None, auto_encoder_attn=False):
    # ft_sizes = [128, 2048]
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    generator=Generator(d_model, tgt_vocab)
    query_embed = [Embeddings(d_model, src_vocab), c(position)]
    tgt_embed = [Embeddings(d_model, tgt_vocab), c(position)]
    query_embed = nn.Sequential(*query_embed)
    tgt_embed = nn.Sequential(*tgt_embed)
    if separate_his_embed:
        his_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    else:
        his_embed = None 
    if separate_cap_embed:
        cap_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    else:
        cap_embed = None 
    cap_encoder = None 
    vid_encoder = None 
    his_encoder = None 
    auto_encoder_generator = None
    auto_encoder_embed = None
    if True:
        if diff_embed:
            auto_encoder_embed = nn.ModuleList()
            for ft_size in ft_sizes:
                embed = [Embeddings(d_model, src_vocab), c(position)]
                auto_encoder_embed.append(nn.Sequential(*embed))
        else:
            auto_encoder_embed = None
        if diff_encoder:
            query_encoder=Encoder(d_model, nb_layers=3 + 2*len(ft_sizes))
        else:
            query_encoder=Encoder(d_model, nb_layers=3 + len(ft_sizes))
        self_attn = nn.ModuleList()
        vid_attn = nn.ModuleList()
        ae_ff = nn.ModuleList()
        vid_encoder=nn.ModuleList()
        auto_encoder_attn_ls = nn.ModuleList()
        for ft_size in ft_sizes:
            ff_layers = [nn.Linear(ft_size, d_model), nn.ReLU(), c(position)]
            vid_encoder.append(nn.Sequential(*ff_layers))
            self_attn.append(c(attn))
            vid_attn.append(c(attn))
            ae_ff.append(c(ff))
            auto_encoder_attn_ls.append(c(attn))
        if diff_gen:
            auto_encoder_generator = nn.ModuleList()
            for ft_size in ft_sizes:
              auto_encoder_generator.append(c(generator))
        else:
            auto_encoder_generator = None
        decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(attn), c(attn), self_attn, vid_attn, auto_encoder_attn_ls, c(ff), ae_ff, dropout), N, ft_sizes)
    else: # query only as source 
        query_encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
    model = EncoderDecoder(
          query_encoder=query_encoder, 
          his_encoder=his_encoder,
          cap_encoder=cap_encoder,
          vid_encoder=vid_encoder,
          decoder=decoder,
          query_embed=query_embed,
          his_embed=his_embed,
          cap_embed=cap_embed,
          tgt_embed=tgt_embed,
          generator=generator,
          auto_encoder_generator=auto_encoder_generator,
          auto_encoder_embed=auto_encoder_embed,
          diff_encoder=diff_encoder,
          auto_encoder_ft=auto_encoder_ft)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model
