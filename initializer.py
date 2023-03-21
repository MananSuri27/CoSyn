from models.plain_model import COSYN
import torch
from geoopt.optim.radam import RiemannianAdam

def initialize_model(num_classes, device, args, socialgraph, params=None):
    if not args:
        data_dir = params["data-dir"]
        x_size = params["x-size"]
        g_size = params["g-size"]
        u_size = params["u-size"]
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
        c = params["c"]
    else:
        data_dir = args.data_dir
        x_size = args.x_size
        g_size = args.g_size
        u_size = args.u_size
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
        c = args.c
  
    model = COSYN(x_size, h_size,g_size, u_size, latent_size, num_classes, dropout, device,  socialgraph, c)

    return model

def initialize_optimizer(type='Adam'):
    if type == 'Adam':
        return torch.optim.Adam
    elif type == 'RiemannianAdam':
        return RiemannianAdam
    return None
