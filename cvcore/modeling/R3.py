import itertools
import math, copy, time

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from .ResidualGCN import ResidualGCN
from .relational_rnn import RelationalMemory
from .mtn import MultiHeadedAttention
from .utils import attention_func

class R3Unit(nn.Module):
    def __init__(self, module_dim = 512):
        super(R3Unit, self).__init__()
        self.module_dim = module_dim
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.relu = nn.ReLU()

        self.reout = nn.Linear(module_dim, module_dim)
        self.combine_linear = nn.Linear(module_dim*2, module_dim)
        # self.bert_linear = nn.Linear(768, module_dim)
        self.obj_linear = nn.Linear(module_dim, module_dim)

        #Attn
        self.question_attention = MultiHeadedAttention(1, module_dim, dropout=0.)
        self.clip_attention = MultiHeadedAttention(1, module_dim, dropout=0.)
        
        # self.bert_linear = nn.Linear(module_dim + 768, module_dim)
        self.object_combined = nn.Linear(module_dim * 3, module_dim)
        
        #GCN---------
        self.GCN = ResidualGCN(module_dim, 0.15)
        self.max_gcn_blocks = 3

        #--------------
        # self.attMem = nn.Linear(module_dim * 2, module_dim)
        self.gru = nn.GRU(input_size=module_dim, hidden_size=module_dim, num_layers=1, batch_first=True, dropout=0.1)
        # self.relation_rnn = RelationalMemory(mem_slots=30, head_size=512, input_size=512, num_heads=1, num_blocks=1, forget_bias=1., input_bias=0.)

    def forward(self, vidObjFeat, nameObjfeat, vecQA, memory=None):
        #z_n
        vidObjFeat = torch.unbind(vidObjFeat, dim = 1)
        attList = list()
        for jdx, objFeat in enumerate(vidObjFeat):
            z_n = attention_func(self.clip_attention, vecQA, objFeat, objFeat)
            q_n = attention_func(self.question_attention, z_n, vecQA, vecQA)
            objFeat = self.object_combined(torch.cat([z_n, q_n, z_n*q_n], dim = -1))
            attList.append(objFeat)
        vidObjFeat = torch.stack(attList, dim = 1)
        
        # nameObjfeat = self.bert_linear(nameObjfeat)
        # attListStack = self.obj_linear(attListStack)
        # attListStack = self.combine_linear(torch.cat((attListStack, nameObjfeat), dim = -1))
        # vidObjFeat = self.elu(attListStack)
        
        # simple OPTION 1
        # if memory is not None:
        #     alpha = self.sigmoid(self.attMem(torch.cat([vidObjFeat, memory], -1)))
        #     memory = (1 - alpha)*vidObjFeat + alpha*memory
        #     vidObjFeat = memory
        # else:
        #     memory = vidObjFeat
        
        # gru OPTION 2
        vidObjFeat = torch.unbind(vidObjFeat, dim = 1)
        objFeats = []
        memory_new = []

        for i, objFeat in enumerate(vidObjFeat):
            if memory is None:
                memory_single = None
            else:
                memory_single = memory[i]
            objFeat, memory_single = self.gru(objFeat.unsqueeze(1), memory_single)
            objFeats.append(objFeat)
            memory_new.append(memory_single)
        vidObjFeat = torch.cat(objFeats, dim = 1)

        # relation rnn OPTION 3
        # if memory is None:
        #     memory = refined_rep
        # refined_rep, memory = self.relation_rnn(refined_rep, memory)
        # refined_rep = refined_rep.view(-1, 30, 512)

        # GCN
        
        # vidObjFeat = F.softmax(self.reout(vidObjFeat), dim = 1)
        # adjacencyMatrix = torch.matmul(vidObjFeat, vidObjFeat.transpose(1, 2))
        vidObjFeat = self.reout(vidObjFeat)
        adjacencyMatrix = torch.matmul(vidObjFeat, vidObjFeat.transpose(1, 2)) / math.sqrt(self.module_dim)
        adjacencyMatrix = F.softmax(adjacencyMatrix, 1)

        for i in range(self.max_gcn_blocks):
            vidObjFeat = self.GCN(vidObjFeat, adjacencyMatrix, i)

        return vidObjFeat, memory_new


    def mask(self, seq, device):
        mask = torch.tensor((seq > 0).float(), device= device)
        ones = torch.ones_like(mask)
        mask = (ones - mask) * (-1e+30)
        return mask