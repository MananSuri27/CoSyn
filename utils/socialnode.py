import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd
import numpy as np

import pickle


def generateEdges(type, number, df):
    path = "/speech/sreyan/aaai/"+type+"_matrix/file"+str(number)

    try:

        with open(path, 'rb') as f:
            sub = pickle.load(f)
        
        for user in sub:
            for follower in sub[user]:
                df.loc[len(df)] = [user,follower]

    except:
        print("Not found: ", path)

    

class Node(DGLDataset):
    def __init__(self):
        super().__init__(name='socialnode')

    def process(self):


        test = pd.read_csv("../Test.csv")
        train = pd.read_csv("../Train.csv")
        val = pd.read_csv("../Val.csv")

        df = pd.concat([train,test,val])

        unique_users = pd.unique(df['user_name'])




        nodes_data = pd.DataFrame(np.delete(unique_users,np.where(unique_users=='nan')), columns=['user_name'])

        edges_data = pd.DataFrame( columns=['src','dest'])

        for i in range(1,45):
            generateEdges("train",i, edges_data)

        for i in range(1,12):
            generateEdges("test",i, edges_data)

        tweet_to_ids = {}

        for ind in nodes_data.index:
            tweet_to_ids.update({nodes_data["user_name"][ind]:ind})

        def mapFunc(a):
            try:
                return tweet_to_ids[a]
            except:
                returnInd = len(tweet_to_ids)
                tweet_to_ids.update({a:returnInd})
                nodes_data.loc[len(nodes_data)] = [a]
                return tweet_to_ids[a]

        
        


        

        edges_data['src'] = edges_data['src'].map(lambda a: mapFunc(a))
        edges_data['dest'] = edges_data['dest'].map(lambda a: mapFunc(a))

        nodes_data['id'] = [i for i in range(len(nodes_data))]

        nodes_data.to_csv("username2id.csv")

        g = torch.empty(size = (len(nodes_data),1024))

        found = 0
        notfound = 0
        for ind in range(len(nodes_data)):
            try:
                try:
                    folder = "Test/"
                    g[ind] = torch.mean(torch.mean(torch.stack(torch.load("../muril/"+folder+nodes_data['user_name'][ind]+".pt")), dim = 0), dim = 0)
                    found = found+1
                except:
                    folder = "Train/"
                    g[ind] = torch.mean(torch.mean(torch.stack(torch.load("../muril/"+folder+nodes_data['user_name'][ind]+".pt")), dim = 0), dim = 0)
                    found = found+1
            except:
                try:
                    try:
                        folder = "Test/" 
                        g[ind] = torch.mean(torch.mean(torch.stack(torch.load("../muril/"+folder+nodes_data['user_name'][ind]+"(1).pt")), dim = 0), dim = 0)
                        found = found+1
                    except:
                        folder = "Train/" 
                        g[ind] = torch.mean(torch.mean(torch.stack(torch.load("../muril/"+folder+nodes_data['user_name'][ind]+"(1).pt")), dim = 0), dim = 0)
                        found = found+1

                except:
                    print("Username not found",nodes_data['user_name'][ind] )
                    g[ind] = torch.normal(mean=torch.zeros(768), std=torch.ones(768))
                    notfound = notfound+1

        print("found",found)
        print("notfound",notfound)

        id = torch.from_numpy(nodes_data['id'].to_numpy())

        edges_dst = torch.from_numpy(edges_data['dest'].to_numpy())
        edges_src = torch.from_numpy(edges_data['src'].to_numpy())


        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape)
        self.graph.ndata['g'] = g.to(dtype=torch.float32)
        self.graph.ndata['id'] = id
        self.graph.ndata['tweet_id'] = id.to(dtype=torch.int64)
        

        # self.graph.edata['weight'] = edge_features

 


    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


