import torch
import torch.nn as nn
import dgl
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import copy
import pickle
from dgl.nn import HGConv

import math

from models.base_model import BaseModel
from models.bchst import CHST

from .hfan import HFAN


class COSYN(BaseModel):
    def __init__(self, x_size, h_size, g_size, u_size,latent_size,num_classes, dropout, device, socialgraph, c=1.0):
        super().__init__()
        self.x_size = x_size
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(2*h_size, h_size//2)
        self.linear2 = nn.Linear(h_size//2+x_size, num_classes)
        self.graph = socialgraph
        self.gcn = HGConv(u_size, g_size, num_heads=1,  bias = False, allow_zero_in_degree=True, activation= torch.sigmoid)
        cell = CHST(x_size, h_size,g_size, device, c)
        self.HFAN = HFAN()


        self.cell = cell
        self.device = device

        self.save_data = []

    def forward(self, batch, h, c, mode='train',save=False):


 



        feats = self.graph.graph.ndata['g']


        feats = self.HFAN(feats)


        res = self.gcn(self.graph.graph, feats)



        ids = batch.ids.to(self.device)



        user = []

        for i in range(len(ids)):
            if ids[i] == len(res):
                ids[i] = ids[i-1] if i!=0 else 0
            user.append(res[ids[i]])


        g = batch.graph.to(self.device)

        embeds = batch.feats.to(self.device)


        user = torch.squeeze(torch.stack(user))







        iou = self.cell.pmath_geo.mobius_matvec(self.cell.W_iou,self.dropout(embeds))
        mso =  self.cell.pmath_geo.mobius_matvec(self.cell.W_mso,self.dropout(user))
        f = self.cell.pmath_geo.mobius_matvec(self.cell.W_f,self.cell.pmath_geo.expmap0(self.dropout(torch.cat((embeds,user), dim = 1))))
        

        g.ndata['iou1'] = iou
        g.ndata['mso1'] = mso
        g.ndata['f'] = f
        g.ndata['h1'] = h
        g.ndata['c1'] = c
        
        dgl.prop_nodes_topo(g, message_func=self.cell.message_func, reduce_func=self.cell.reduce_func, apply_node_func=self.cell.apply_node_func)

        h1 = self.cell.pmath_geo.logmap0(g.ndata.pop('h1'))


        g.ndata['iou1'] = iou
        g.ndata['mso1'] = mso
        g.ndata['f'] = f
        g.ndata['h1'] = h
        g.ndata['c1'] = c
        
        
        dgl.prop_nodes_topo(g, message_func=self.cell.message_func, reduce_func=self.cell.reduce_func, apply_node_func=self.cell.apply_node_func,reverse=True)



        h2 = self.cell.pmath_geo.logmap0(g.ndata.pop('h1'))


        h = self.dropout(torch.cat((h1,h2), dim=1))
        h = self.linear1(h)
        
        if mode=='train':
            mask = batch.train_mask
        elif mode=='val':
            mask = batch.val_mask
        elif mode == 'test':
            mask = batch.test_mask

        if save:
            g.ndata['embs'] = h
        
        h = h[mask==1]

        root_feat = batch.feats[mask==1]

        layer = torch.cat((root_feat,h),dim=-1)
        logits = self.linear2(layer)

        if save:
            self.save_data.append((g.cpu(),logits,g.ndata['y']))

        return logits
        
    def save_embs(self):
        with open("embeddings.pkl",'wb') as f:
            pickle.dump(self.save_data,f)