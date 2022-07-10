import itertools
import math, copy, time
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from transformers import BertTokenizer, BertModel

from .R3 import R3Unit
from .mtn import MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding, \
    Generator, Embeddings, Encoder, EncoderLayer, Decoder, DecoderLayer, DecoderLayerSimple, \
    DecoderSimple, PointerGenerator
from .utils import init_modules, subsequent_mask, LabelSmoothing, SimpleLossCompute, attention_func


class COST(nn.Module):
    def __init__(self, cfg, vocab):
        super().__init__()
        # assert len(vocab['question_token_to_idx']) == len(vocab['answer_token_to_idx'])
        print(f"COST config: N({cfg.NUM_BLOCKS}) h({cfg.NUM_HEADS}) d_ff({cfg.D_FF}) dropout({cfg.DROPOUT}) \
                AutoEncoder: {cfg.AE_USE}")

        c = copy.deepcopy
        attn = MultiHeadedAttention(cfg.NUM_HEADS, cfg.D_MODEL)
        ff = PositionwiseFeedForward(cfg.D_MODEL, cfg.D_FF, cfg.DROPOUT)
        position = PositionalEncoding(cfg.D_MODEL, cfg.DROPOUT)
        r3_unit = R3Unit(cfg.D_MODEL)
        
        self.len_vocab = len(vocab['question_token_to_idx'])
        self.vocab = vocab['question_token_to_idx']
        self.pad_id = vocab['question_token_to_idx']['<NULL>']
        self.start_id = vocab['question_token_to_idx']['<SOS>']
        self.unk_id = vocab['question_token_to_idx']['<UNK>']
        self.end_id = vocab['question_token_to_idx']['<EOS>']
        self.ae_use = cfg.AE_USE
        self.d_model = cfg.D_MODEL

        # self.bertbase = BertModel.from_pretrained("bert-base-uncased")
        self.linguistic_encoder = Encoder([cfg.D_MODEL for i in range(4)])
        self.visual_encoder = Encoder([cfg.D_MODEL, cfg.D_MODEL])
        # self.bert_encoder = Encoder([768])
        self.decoder = Decoder(DecoderLayer(cfg.D_MODEL, c(attn), c(attn), c(ff), #main decoder
                                            # c(attn), c(attn),
                                            c(attn),
                                            # c(attn), c(attn), c(ff), c(attn),  #ae decoder 2
                                            c(attn), c(attn), c(ff), c(attn),  #ae decoder
                                            cfg.DROPOUT), cfg.NUM_BLOCKS, [cfg.D_MODEL, cfg.D_MODEL])
        ques_embed = [Embeddings(cfg.D_MODEL, self.len_vocab), c(position)]
        # tgt_embed = [Embeddings(cfg.D_MODEL, self.len_vocab), c(position)]
        self.ques_embed = nn.Sequential(*ques_embed)
        # self.tgt_embed = nn.Sequential(*tgt_embed)
        self.tgt_embed = self.ques_embed 
        # self.generator=Generator(cfg.D_MODEL, self.len_vocab, self.tgt_embed[0].lut.weight)
        pointer_attn = MultiHeadedAttention(1, cfg.D_MODEL, dropout=0)
        self.generator = PointerGenerator(cfg.D_MODEL, self.tgt_embed[0].lut.weight, pointer_attn)
        self.ae_generator= Generator(cfg.D_MODEL, self.len_vocab, self.tgt_embed[0].lut.weight)

        #Video
        # app_feat_proj = [nn.Linear(cfg.APP_FEAT, cfg.D_MODEL), nn.ReLU(), c(position)]
        # self.app_feat_proj = nn.Sequential(*app_feat_proj)
        motion_feat_proj = [nn.Linear(cfg.APP_FEAT, cfg.D_MODEL), nn.ReLU(), c(position)]
        self.motion_feat_proj = nn.Sequential(*motion_feat_proj)

        self.obj_app_feat_proj = nn.Linear(cfg.OBJ_APP_FEAT, cfg.D_MODEL)
        self.obj_spa_feat_proj = nn.Linear(7, cfg.D_MODEL)

        self.r3 = c(r3_unit)
        self.past_reasoning = DecoderSimple(DecoderLayerSimple(cfg.D_MODEL, c(attn), c(attn), c(ff), #main decoder
                                            cfg.DROPOUT), cfg.NUM_BLOCKS, [])
        self.out_vid_proj = nn.Linear(cfg.D_MODEL*2, cfg.D_MODEL)
        self.dialogs_position = Embeddings(cfg.D_MODEL*2, 10)
        self.dialogs_proj = nn.Linear(cfg.D_MODEL*2, cfg.D_MODEL)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

        #Loss
        self.loss_l = 1
        self.criterion = LabelSmoothing(size=self.len_vocab, padding_idx=self.pad_id, smoothing=0.1)
        self.loss_compute = SimpleLossCompute(self.generator, self.ae_generator, self.criterion, self.loss_l)

        # Attn
        self.motion_attention = MultiHeadedAttention(1, cfg.D_MODEL, dropout=0.)
        self.answer_attention = MultiHeadedAttention(1, cfg.D_MODEL, dropout=0.)

        self.agg_motion_linear = nn.Linear(cfg.D_MODEL*2, cfg.D_MODEL)

        init_modules(self.modules(), w_init=cfg.TRAIN_WEIGHT_INIT)

    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(
            np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
        return torch.index_select(a, dim, order_index)

    def encode(self, question, dialog, tgt,
            obj_app_feat, obj_spatial_feat, 
            obj_name_enc, obj_name_ids,
            appearance_feat, motion_feat):
        ques_mask = (question != self.pad_id).unsqueeze(-2)
        ques_encoded, ae_encoded, ae_encoded_2, dialog_encoded, dialog_encoded_mask, \
            tgt_embeds, tgt_attn_mask, (b, n_d, n_s) = self.linguistic_pre_encode(question, dialog, tgt)
        ques_encoded, ae_encoded, ae_encoded_2, dialog_encoded  = self.linguistic_encoder([ques_encoded, ae_encoded, ae_encoded_2, dialog_encoded])

        past_questions_encoded = dialog_encoded.view(b, n_d, n_s, -1)[:, 1::2]
        past_answers_encoded = dialog_encoded.view(b, n_d, n_s, -1)[:, 0::2]
        
        # obj_name_feat = self.bertbase(obj_name_enc).last_hidden_state[0][obj_name_ids]
        # obj_name_feat = self.bert_encoder([obj_name_feat])[0]
        obj_name_feat = None
        
        obj_feat_encoded, motion_feat_encoded, motion_feat_mask = self.visual_pre_encode(obj_app_feat, obj_spatial_feat, appearance_feat, motion_feat)
        obj_feat_encoded, motion_feat_encoded = self.visual_encoder([obj_feat_encoded, motion_feat_encoded])
        
        ############ 

        # Augment with global context
        agg_motion_feat = attention_func(self.motion_attention, ques_encoded, motion_feat_encoded, motion_feat_encoded, motion_feat_mask)
        agg_motion_feat = agg_motion_feat.unsqueeze(dim=1)
        agg_motion_feat = self.tile(agg_motion_feat, 1, 30)

        # Past reasoning
        previous_memory = None
        past_r3_outs = []
        for i in range(9):
            out, previous_memory = self.r3(obj_feat_encoded, obj_name_feat, past_questions_encoded[:, i], previous_memory)
            # Augment with global context
            out = torch.cat((agg_motion_feat, out), dim=-1)
            out = self.agg_motion_linear(out)
            out = self.elu(out)
            past_r3_outs.append(out)

        current_r3_out, previous_memory = self.r3(obj_feat_encoded, obj_name_feat, ques_encoded, previous_memory)
        # Augment with global context
        current_r3_out = torch.cat((agg_motion_feat, current_r3_out), dim=-1)
        current_r3_out = self.agg_motion_linear(current_r3_out)
        current_r3_out = self.elu(current_r3_out)


        past_answers_encoded_1 = [attention_func(self.answer_attention, past_r3_outs[i], past_answers_encoded[:, i], past_answers_encoded[:, i]).unsqueeze(1).tile((1,30,1)) for i in range(9)]
        past_r3_outs = [torch.cat([past_r3_outs[i], past_answers_encoded_1[i]], -1) for i in range(9)]
        past_r3_outs = [past_r3_outs[i] + self.dialogs_position( (torch.ones(past_r3_outs[i].shape[:-1])*i).to(dtype=question.dtype, device=question.device) ) for i in range(9)]
        past_r3_outs = torch.cat([self.dialogs_proj(past_r3_outs[i]).unsqueeze(2) for i in range(9)], dim=2)
        

        current_r3_out = torch.cat([current_r3_out, attention_func(self.answer_attention, current_r3_out, past_answers_encoded[:, -1], past_answers_encoded[:, -1]).unsqueeze(1).tile((1,30,1))], -1)
        current_r3_out = current_r3_out + self.dialogs_position( (torch.ones(current_r3_out.shape[:-1])*9).to(dtype=question.dtype, device=question.device) )       
        current_r3_out = self.dialogs_proj(current_r3_out)

        past_r3_outs = torch.cat([past_r3_outs, current_r3_out.unsqueeze(2)], dim=2)

        out_vid = [self.past_reasoning(current_r3_out[:, i].unsqueeze(1), None, past_r3_outs[:, i], None) \
                                                                                               for i in range(30)]
        out_vid = torch.cat(out_vid, 1)
        out_vid = torch.cat([out_vid, current_r3_out], -1)
        out_vid = self.out_vid_proj(out_vid)  

        vid_mask = None

        return [dialog_encoded, ques_encoded, out_vid, motion_feat_encoded], [dialog_encoded_mask, ques_mask, vid_mask, motion_feat_mask], [ae_encoded, ae_encoded_2], tgt_embeds, tgt_attn_mask
        # return [ques_encoded, motion_feat_encoded, out_vid], [ques_mask, motion_feat_mask, vid_mask], [ae_encoded]

    def forward_features(self, encoder_out_list, encoder_pad_mask_list, tgt_embeds, tgt_attn_mask, ae_use, ae_fts):
        out, out_ae = self.decode(encoder_out_list=encoder_out_list,
                        encoder_pad_mask_list=encoder_pad_mask_list,
                        tgt_embeds=tgt_embeds, tgt_attn_mask=tgt_attn_mask, 
                        ae_use=ae_use, ae_ft = ae_fts)
        return out, out_ae

    def forward(self, batch):
        '''
        0video_idx, 1answers_in, 2answers_out, 3answers_len, 
        4questions, 5question_len, 
        6dialog, 7dialog_len, 
        8obj_app_feat, 9obj_spatial_feat, 10obj_name_enc, 11obj_name_ids, 
        12appearance_feat, 13motion_feat = batch
        '''
        encoder_out_list, encoder_pad_mask_list, ae_fts, tgt_embeds, tgt_attn_mask = self.encode(batch[4], batch[6], batch[1],\
                                            batch[8], batch[9], \
                                            batch[10], batch[11], \
                                            batch[12], batch[13])

        outputs, outputs_ae = self.forward_features(encoder_out_list, encoder_pad_mask_list, tgt_embeds, tgt_attn_mask, self.ae_use, ae_fts)
        losses_dict = {}
        loss_seq_ans, loss_seq_ques = self.loss_compute(outputs, batch[2], batch[3],
                        outputs_ae, batch[4], batch[5], batch[4], encoder_out_list[1], encoder_pad_mask_list[1], tgt_embeds)
        losses_dict.update({"anwser loss": loss_seq_ans, "ques loss": loss_seq_ques})
        return losses_dict

    def linguistic_pre_encode(self, ques, dialogs, tgt):
        b, n_d, n_s = dialogs.shape
        dialogs = dialogs.view(b, n_d*n_s)
        dialog_enconded = self.ques_embed(dialogs)
        dialog_encoded_mask = (dialogs != self.pad_id).unsqueeze(-2)
        ae_encoded = self.ques_embed(ques)
        ae_encoded_2 = self.ques_embed(ques)
        ques_encoded = self.ques_embed(ques)

        tgt_attn_mask = (tgt != self.pad_id).unsqueeze(-2)
        tgt_attn_mask = tgt_attn_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_attn_mask.data))
        tgt_embeds = self.tgt_embed(tgt)
        return ques_encoded, ae_encoded, ae_encoded_2, dialog_enconded, dialog_encoded_mask, tgt_embeds, tgt_attn_mask, (b, n_d, n_s)
        # return self.ques_embed(ques), dialog_enconded, dialog_encoded_mask, (b, n_d, n_s)

    def visual_pre_encode(self, obj_app, obj_spa, appearance_feat, motion_feat):
        # obj_app = obj_app.mean(3).transpose(1,2)
        # obj_spa = obj_spa.mean(3).transpose(1,2)
        # appearance_feat = appearance_feat.mean(2)
        
        b, c, o, f, _ = obj_app.shape
        obj_app = obj_app.view(b, o, c*f, -1)
        obj_spa = obj_spa.view(b, o, c*f, -1)
        # appearance_feat = appearance_feat.view(b, c*f, -1)

        obj = self.tanh(self.obj_app_feat_proj(obj_app)) * \
                self.sigmoid(self.obj_spa_feat_proj(obj_spa))

        # appearance_feat = self.app_feat_proj(appearance_feat)
        motion_feat_mask = (torch.sum(motion_feat != 1, dim=2) != 0).unsqueeze(-2)
        motion_feat =  motion_feat * motion_feat_mask.squeeze().unsqueeze(-1).expand_as(motion_feat).float()
        motion_feat = self.motion_feat_proj(motion_feat)
        motion_feat = self.relu(motion_feat)
        # return obj, appearance_feat, motion_feat
        return obj, motion_feat, motion_feat_mask

    def decode(self, encoder_out_list, encoder_pad_mask_list, 
                    tgt_embeds, tgt_attn_mask, ae_use, ae_ft):
        #batch, n_seq, ff_dim
        out, out_ae = self.decoder(tgt_embeds, tgt_attn_mask,
                    encoder_out_list, encoder_pad_mask_list, ae_use, ae_ft)
        return out, out_ae