import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd


class Node(DGLDataset):
    def __init__(self,id,type):
        self.id = id
        self.type = type
        super().__init__(name='node')

    def process(self):
        nodes_data = pd.read_csv('./members/'+self.type+"/"+self.id+".csv")
        edges_data = pd.read_csv('./interactions/'+self.type+"/"+self.id+".csv")

        nodes_data['x'] = nodes_data['x'].map(str)

        tweet_to_ids = {}

        for ind in nodes_data.index:
            tweet_to_ids.update({nodes_data["index"][ind]:ind})

        edges_data['src'] = edges_data['src'].map(lambda a: tweet_to_ids[a])
        edges_data['dest'] = edges_data['dest'].map(lambda a: tweet_to_ids[a])

        x = torch.empty(size = (len(nodes_data),1024))
        # g = torch.empty(size = (len(nodes_data),768))

        for ind in range(len(nodes_data)):
            x[ind] = torch.load("embeds/"+nodes_data['x'][ind]+".pt")



        userID = pd.read_csv('username2id.csv')

        def mapU2ID(a):
            # if a=="nan":
            #     return(len(userID))
            if len(userID.loc[userID["user_name"]==a]["id"].values)==0:
                return len(userID)
            return  userID.loc[userID["user_name"]==a]["id"].values[0]

        y = torch.from_numpy(nodes_data['y'].to_numpy())
        del_t = torch.from_numpy(nodes_data['del_t'].to_numpy())
        train_mask = torch.from_numpy(nodes_data['train_mask'].to_numpy())
        val_mask = torch.from_numpy(nodes_data['val_mask'].to_numpy())
        test_mask = torch.from_numpy(nodes_data['test_mask'].to_numpy())
        id = torch.from_numpy((nodes_data['u_name'].map(lambda a: mapU2ID(a))).to_numpy())
        tweet_id = torch.from_numpy(nodes_data['x'].astype("int64").to_numpy())



        edges_dst = torch.from_numpy(edges_data['dest'].to_numpy())
        edges_src = torch.from_numpy(edges_data['src'].to_numpy())

        print(self.id, self.type, nodes_data.shape[0])

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
        self.graph.ndata['x'] = x.to(dtype=torch.float32)
        # self.graph.ndata['g'] = g.to(dtype=torch.float32)
        self.graph.ndata['y'] = y
        self.graph.ndata['del_t'] = del_t.to(dtype=torch.float32)
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['test_mask'] = test_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['id'] = id
        self.graph.ndata['tweet_id'] = tweet_id.to(dtype=torch.int64)
        


    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
