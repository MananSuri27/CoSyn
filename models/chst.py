import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import copy
import math
import itertools
from geoopt.manifolds.poincare import PoincareBall

from models.attn_layers import HyperAttn, Attention

class CHST(torch.nn.Module):
    def __init__(self, x_size, h_size,g_size, device, c):
        super(CHST, self).__init__()
        self.W_iou = nn.Parameter(torch.Tensor(2 * h_size, x_size))
        self.U_iou = nn.Parameter(torch.Tensor(2 * h_size, h_size))
        self.W_mso = nn.Parameter(torch.Tensor(3 * h_size, g_size))
        self.U_mso = nn.Parameter(torch.Tensor(3 * h_size, h_size))
        self.b_mso = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.b_iou = nn.Parameter(torch.zeros(1, 2 * h_size))
        self.W_f = nn.Parameter(torch.Tensor(h_size, x_size +g_size))
        self.U_f = nn.Parameter(torch.Tensor(h_size, h_size))
        self.b_f = nn.Parameter(torch.zeros(1, h_size))
        self.W_q = nn.Parameter(torch.Tensor(h_size, x_size))
        self.b_q = nn.Parameter(torch.zeros(1, h_size))
        self.W_k = nn.Parameter(torch.Tensor(h_size, h_size))
        self.b_k = nn.Parameter(torch.zeros(1, h_size))
        self.W_c = nn.Parameter(torch.Tensor(h_size, h_size))
        self.b_c = nn.Parameter(torch.zeros(1,h_size))
        self.temp = torch.tensor([1e-6]).to(device)
        self.const_bias_param = torch.nn.Parameter(torch.Tensor(5))
        self.device = device
        self.x_size = x_size
        self.h_size = h_size
        self.reset_parameters()
        self.pmath_geo = PoincareBall(c=c)
        self.attn = HyperAttn()
        self.b = nn.Parameter(torch.Tensor([1e-6]), requires_grad=True)
        self.a = nn.Parameter(torch.Tensor([0.02]), requires_grad=True)
        self.haw_1 = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.haw_2 = nn.Parameter(torch.Tensor([1]), requires_grad=True)


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.h_size)
        for weight in itertools.chain.from_iterable([self.W_iou, self.U_iou, self.W_f, self.U_f, self.W_q, self.W_k, self.W_c, self.U_mso, self.W_mso]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def message_func(self, edges):
        return {'h1': edges.src['h1'], 'c1': edges.src['c1'], 'del_t': edges.src['del_t'], 'x1':edges.src['x1']}

    def reduce_func(self, nodes):
        g_t = self.b*torch.exp(-self.a*nodes.mailbox['del_t'])
        x_ = self.pmath_geo.mobius_add(self.pmath_geo.mobius_matvec(self.W_q,self.pmath_geo.expmap0(nodes.data['x'])),self.b_q)        
        h_ = self.pmath_geo.mobius_add(self.pmath_geo.mobius_matvec(self.W_k,nodes.mailbox['h1']),self.b_k)
        scaled_scores = self.attn(h_, x_, g_t, self.pmath_geo)
        h_scaled = self.pmath_geo.mobius_pointwise_mul(scaled_scores.unsqueeze(dim=-1), nodes.mailbox['h1'])
        h_tild = self.pmath_geo.weighted_midpoint(h_scaled, reducedim=[1])
        c_k = nodes.mailbox['c1']
        c_sk = self.pmath_geo.expmap0(torch.tanh(self.pmath_geo.logmap0(self.pmath_geo.mobius_add(self.pmath_geo.mobius_matvec(self.W_c,c_k),self.b_c))))
        c_sk_hat = self.pmath_geo.mobius_pointwise_mul(c_sk,g_t.unsqueeze(dim=-1))
        c_Tk = self.pmath_geo.mobius_add(-c_sk,c_k)
        c_k_tilde = self.pmath_geo.mobius_add(c_Tk,c_sk_hat)
        temp = self.pmath_geo.mobius_matvec(self.U_f, nodes.mailbox['h1'])
        f =  torch.sigmoid(self.pmath_geo.logmap0((self.pmath_geo.mobius_add(nodes.data['f'].unsqueeze(1).repeat(1,temp.shape[1],1), temp ))))   
        c = self.pmath_geo.weighted_midpoint(self.pmath_geo.mobius_pointwise_mul(f,c_k_tilde), reducedim=[1])
        x = torch.mean(nodes.mailbox['x1'],dim=1)

        return {'iou1': self.pmath_geo.mobius_add(nodes.data['iou1'], self.pmath_geo.mobius_matvec(self.U_iou, h_tild)), 'mso1': self.pmath_geo.mobius_add(nodes.data['mso1'], self.pmath_geo.mobius_matvec(self.U_mso, h_tild)),'c1': c, 'x1':x}

    def apply_node_func(self, nodes):
        x = torch.add(nodes.data['x1'],nodes.data['x'])/2
        iou = self.pmath_geo.mobius_add(nodes.data['iou1'], self.b_iou)
        mso = self.pmath_geo.mobius_add(nodes.data['mso1'], self.b_mso)
        i,  u , = torch.chunk(iou, 2, 1)
        m, s, o = torch.chunk(mso, 3,1)
        i, o, u , m ,s= torch.sigmoid(self.pmath_geo.logmap0(i)), torch.sigmoid(self.pmath_geo.logmap0(o)), torch.tanh(self.pmath_geo.logmap0(u)), torch.sigmoid(self.pmath_geo.logmap0(m)), torch.sigmoid(self.pmath_geo.logmap0(s))
        c = self.pmath_geo.mobius_add(self.pmath_geo.mobius_add(self.pmath_geo.mobius_pointwise_mul(i,u),nodes.data['c1']), self.pmath_geo.mobius_pointwise_mul(m,s))
        h = self.pmath_geo.mobius_pointwise_mul(o,torch.tanh(self.pmath_geo.logmap0(c)))
        return {'h1': h, 'c1': c, 'x1':x}


