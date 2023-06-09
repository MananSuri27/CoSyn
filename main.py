from collections import namedtuple
import argparse
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import time
import os
from datetime import datetime
import pickle
import gc
import copy

import torch as th
import dgl
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

from initializer import initialize_model,initialize_optimizer
from dataset import TwitterDataset, batcher, create_dataset
from loss import loss_fn

import pandas as pd
import collections


import random
import numpy as np

seed = 7
th.cuda.manual_seed(seed)
th.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

df = pd.read_csv("")
id2label= collections.defaultdict(int)
id2hof = collections.defaultdict(int)
id2reply= collections.defaultdict(int)
id2comment= collections.defaultdict(int)
id2parent= collections.defaultdict(int)

for i in range(len(df)):
    id = df["tweet_id"][i]
    

for i in range(len(df)):
    id = df["tweet_id"][i]
    if df["label"][i] == "HOF":
        id2hof[id] = 1
        if df["annotation"][i] =="I":
            id2label[id] = 1
        else:
            id2label[id] = 0
    else:
        id2hof[id] = 0 

    if df["type"][i] == "reply":
        id2reply[id] = 1
    else:
        id2reply[id] = 0 
    
    if df["type"][i] == "comment":
        id2comment[id] = 1
    else:
        id2comment[id] = 0 
    
    if df["type"][i] == "parent":
        id2parent[id] = 1
    else:
        id2parent[id] = 0 


def train_loop(model,data_loader,optimizer,device,h_size,beta,gamma):
    train_preds = []
    train_true_l = []
    train_logits = []

    model.train()
    for step, batch in tqdm(enumerate(data_loader),total=len(data_loader)):
        g = batch.graph
        n = g.number_of_nodes()

        h = th.zeros((n, h_size)).to(device)
        c = th.zeros((n, h_size)).to(device)

        logits = model(batch, h, c, 'train')

        true_labels = batch.label[batch.train_mask==1]

        loss = loss_fn(logits, true_labels, 2, true_labels.unique(return_counts=True)[1].tolist(), device, beta, gamma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = th.argmax(logits, 1)

        train_logits.append(logits)
        train_preds.extend(pred.to('cpu'))
        train_true_l.extend(true_labels.to('cpu'))

    train_metrics = model.compute_metrics(train_true_l,train_preds)
    train_logits = th.cat(train_logits).to(device)
    train_true_l = th.tensor(train_true_l).to(device)
    train_loss = loss_fn(train_logits, train_true_l, 2, train_true_l.unique(return_counts=True)[1].tolist(), device, beta, gamma)
    train_metrics["loss"] = train_loss.item()

    print("Train (Loss {:.4f} | M.F1 {:.4f} | Rec {:.4f} |".format(train_metrics["loss"], train_metrics["f1"], train_metrics["recall"]))


def val_loop(model,data_loader,device,h_size,beta,gamma):
    val_preds = []
    val_true_l = []
    val_logits = []
    model.eval()
    for step, batch in enumerate(data_loader):
        g = batch.graph
        n = g.number_of_nodes()

        h = th.zeros((n, h_size)).to(device)
        c = th.zeros((n, h_size)).to(device)

        logits = model(batch, h, c, 'val')
        val_logits.append(logits)
        true_labels = batch.label[batch.val_mask==1]
        pred = th.argmax(logits, 1)
        val_preds.extend(pred.to('cpu'))
        val_true_l.extend(true_labels.to('cpu'))

    val_metrics = model.compute_metrics(val_true_l,val_preds)
    val_logits = th.cat(val_logits).to(device)
    val_true_l = th.tensor(val_true_l).to(device)
    val_loss = loss_fn(val_logits, val_true_l, 2, val_true_l.unique(return_counts=True)[1].tolist(), device, beta, gamma)
    val_metrics["loss"] = val_loss.item()

    print("Val (Loss {:.4f} | M.F1 {:.4f} | Rec {:.4f} |".format(val_metrics["loss"], val_metrics["f1"], val_metrics["recall"]))

    return val_metrics

def test_loop(model,data_loader,device,h_size,beta,gamma,save=False):
    test_preds = []
    test_true_l = []
    test_logits = []

    id_test = []
    implicit =[]
    implicit_pred =[]

    implicit_reply = []
    implicit_reply_pred = []

    implicit_comment = []
    implicit_comment_pred = []

    model.eval()
    for step, batch in enumerate(data_loader):
        g = batch.graph
        n = g.number_of_nodes()

        h = th.zeros((n, h_size)).to(device)
        c = th.zeros((n, h_size)).to(device)

        logits = model(batch, h, c, 'test',save=save)
        #logits = logits[batch.test_mask==1]
        test_logits.append(logits)
        true_labels = batch.label[batch.test_mask==1]
        pred = th.argmax(logits, 1)
        test_preds.extend(pred.to('cpu'))
        
        test_true_l.extend(true_labels.to('cpu'))

        
        Ids = list(g.ndata["tweet_id"])
        for i in range(n):
            
            id = int(Ids[i].item())
            id_test.append(id)

           # print(id)
            if id2label[id] ==1:       
                implicit.append(id2label[id])
                implicit_pred.append(int(pred[i].to("cpu")))
        
        for i in range(n):
            id = int(Ids[i].item())
            #print(id)
            if id2label[id] ==1 and id2reply[id]==1:       
                implicit_reply.append(id2label[id])
                implicit_reply_pred.append(int(pred[i].to("cpu")))
        
        for i in range(n):
            id = int(Ids[i].item())
            #print(id)
            if id2label[id] ==1 and id2comment[id]==1:       
                implicit_comment.append(id2label[id])
                implicit_comment_pred.append(int(pred[i].to("cpu")))
        

    if save:
        model.save_embs()
        model.save_data = []
        gc.collect()

    print(len(implicit), len(implicit_pred))


    implicit_metrics = model.compute_metrics(implicit,implicit_pred)

    print("implicit", implicit_metrics)

    implicit_reply_metrics = model.compute_metrics(implicit_reply,implicit_reply_pred)

    print("implicit_reply", implicit_reply_metrics)

    implicit_comment_metrics = model.compute_metrics(implicit_comment,implicit_comment_pred)

    print("implicit_comment", implicit_comment_metrics)


    pred_df = pd.DataFrame(list(zip(id_test, test_preds)), columns =['id', 'pred'])

    
    
    test_metrics = model.compute_metrics(test_true_l,test_preds)
    

    
    test_logits = th.cat(test_logits).to(device)
    test_true_l = th.tensor(test_true_l).to(device)
    test_loss = loss_fn(test_logits, test_true_l, 2, test_true_l.unique(return_counts=True)[1].tolist(), device, beta, gamma)
    test_metrics["loss"] = test_loss.item()
    
    print("Test (Loss {:.4f} | M.F1 {:.4f} | Rec {:.4f} |".format(test_metrics["loss"], test_metrics["f1"], test_metrics["recall"]))
    return test_metrics, pred_df

def main(args, params=None):
    start = time.time()

    if not args:
        data_dir = params["data-dir"]
        x_size = params["x-size"]
        g_size = params["g-size"]
        h_size = params["h-size"]
        latent_size = params["latent_size"]
        dropout = params["dropout"]
        lr = params["lr"]
        weight_decay = params["weight-decay"]
        epochs = params["epochs"]
        beta = params["beta"]
        gamma = params["gamma"]
        batch_size = params["batch-size"]
        patience = params["patience"]
        min_epochs = params["min-epochs"]
        device = params["device"]
        optim_type = params["optimizer"]
        save = params["save"]
        save_dir = params["save-dir"]
        print(params)
    else:
        data_dir = args.data_dir
        x_size = args.x_size
        g_size = args.g_size
        h_size = args.h_size
        latent_size = args.latent_size
        dropout = args.dropout
        lr = args.lr
        weight_decay = args.weight_decay
        epochs = args.epochs
        beta = args.beta
        gamma = args.gamma
        batch_size = args.batch_size
        patience = args.patience
        min_epochs = args.min_epochs
        device = args.device
        optim_type = args.optimizer
        save = args.save
        save_dir = args.save_dir
        print(args)


    train_dataset, val_dataset, test_dataset = create_dataset(data_dir)
    train_trees = train_dataset.trees
    val_trees = val_dataset.trees
    test_trees = test_dataset.trees

    num_classes = train_dataset.num_classes

    if device=='auto':
        # device = 'cuda' if th.cuda.is_available() else 'cpu'
        device = 'cpu'



    print("Device:",device)

    with open('./data/social_network.pkl', 'rb') as f:
        socialgraph = pickle.load(f)

    if not args:
        model = initialize_model(num_classes, device, None, socialgraph, params)
    else:
        model = initialize_model(num_classes, device, args, socialgraph )
    model.to(device)
    print(model)

    optimizer = initialize_optimizer(optim_type)(model.parameters(),lr=lr,weight_decay=weight_decay)

    train_loader = DataLoader(dataset=train_trees,batch_size=batch_size,collate_fn=batcher(device),shuffle=False,num_workers=0)
    val_loader = DataLoader(dataset=val_trees,batch_size=batch_size,collate_fn=batcher(device),shuffle=False,num_workers=0)
    test_loader = DataLoader(dataset=test_trees,batch_size=batch_size,collate_fn=batcher(device),shuffle=False,num_workers=0)

    counter=0
    best_val_metrics = model.init_metric_dict()
    test_metrics = None
    pred_df = pd.DataFrame()
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        print("Epoch: ",epoch)
        train_loop(model,train_loader,optimizer,device,h_size,beta,gamma)

        val_metrics = val_loop(model,val_loader,device,h_size,beta,gamma)
        if model.has_improved(best_val_metrics, val_metrics):
            test_metrics, pred_df = test_loop(model,test_loader,device,h_size,beta,gamma)
            best_val_metrics = val_metrics
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter == patience and epoch > min_epochs:
                print("Early stopping")
                break

    print("(Loss {:.4f} | M.F1 {:.4f} | Rec {:.4f} |".format(test_metrics["loss"], test_metrics["f1"], test_metrics["recall"]))
    print(test_metrics["conf_mat"])

    if not os.path.exists('results'):
        os.makedirs('results')

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    end = time.time()
    print("Time Elapsed: ",end-start)

    save_dict = {}
    save_dict['test_metrics'] = test_metrics
    save_dict['args'] = args
    save_dict['params'] = params
    save_dict['time'] = current_time
    

    dir_ = './data'
    save =True
    fname = dir_ + "/" + current_time + ".pkl"

    pred_df.to_csv(dir_ + "/preds" + str(test_metrics['f1'])+".csv")

    with open(fname,'wb') as f:
        pickle.dump(save_dict,f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto',choices=['auto','cpu','cuda'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--x-size', type=int, default=768)
    parser.add_argument('--g-size', type=int, default=768)

    parser.add_argument('--u-size', type=int, default = 768)
    parser.add_argument('--latent-size', type=int, default = 512)

    parser.add_argument('--h-size', type=int, default=768)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--patience', type=int, default=60)
    parser.add_argument('--min-epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0013086417263456422)
    parser.add_argument('--weight-decay', type=float, default=0.0003053617595661161)
    parser.add_argument('--dropout', type=float, default=0.41789978091535973)
    parser.add_argument('-beta', '--beta', default=0.999999, type=float)
    parser.add_argument('-gamma', '--gamma', default=1.7269739598697345, type=float)
    parser.add_argument('--data-dir', type=str, default='./data',help='directory for data')
    parser.add_argument('--optimizer', type=str, default='Adam',choices=['Adam','RiemannianAdam'])
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save-dir', type=str, default='res',help='save directory')
    parser.add_argument('--c', type=float, default=1.0)

    args = parser.parse_args()

    main(args)