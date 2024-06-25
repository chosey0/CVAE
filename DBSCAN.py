from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from models.CVAE.cvae import CVAE
from Ztemp.criterion import likelihood_loss

from torch.utils.data import DataLoader

from utils.training.dataset import eBestDataset, eBestMultiDataset
from utils.training.collate import collate, multi_collate

from Ztemp.init_dataset import load_dataset
from tqdm import tqdm

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output
    
def main():
    test = ['/다보링크(340360)/20240422-20240503', 
     '/한국석유(004090)/20240422-20240503', 
     '/태성(323280)/20240422-20240503', 
     '/한글과컴퓨터(030520)/20240422-20240503', 
     '/본느(226340)/20240422-20240503', 
     '/코스나인(082660)/20240422-20240503']
    
    with pd.HDFStore("Base.h5", "r") as store:
        test_set = load_dataset(test, store, resample=False)
        
    # train_loaders = eBestMultiDataset([DataLoader(eBestDataset(subset, key), batch_size=1, collate_fn=collate, shuffle=True) for key, subset, _ in test_set])
    # loader = DataLoader(train_loaders, collate_fn=multi_collate, batch_size=len(train_loaders.subsets))
    
    loader = DataLoader(eBestDataset(test_set[0][1],test_set[0][0]), batch_size=1, collate_fn=collate, shuffle=True)
    
    # while True:
    #     item = next(loader)
    
    model = CVAE(latent_dim=10, num_param=2, in_window_size=512, out_window_size=512, scale_flag=1)
    model.load_state_dict(torch.load("Result/temp/model_1.pt"))
    
    criterion = likelihood_loss(0.001, "MSE")
    criterion.load_state_dict(torch.load("Result/temp/criterion_1.pt"))
    
    model.to("cuda")
    criterion.to("cuda")
    
    latent_vecs = []
    mu_vecs = []
    log_var_vecs = []
    real_results = []
    
    loss = []
    with torch.no_grad():
        model.eval()
        # reals는 row 1개
        for batch_step, (item, reals) in tqdm(enumerate(loader), desc="Test"):
            item = item.to("cuda")
            
            z, mu, log_var, rec = model(item)
            
            L, rec_loss, kl, likelihood, elbo = criterion(rec, item, model.kl_div)
            
            latent_vecs.append(z)
            mu_vecs.append(mu)
            log_var_vecs.append(log_var)
            
            reals["model_vol"] = rec.squeeze(-1).detach().cpu().numpy()[:, 0].flatten()
            reals["model_amount"] = rec.squeeze(-1).detach().cpu().numpy()[:, 1].flatten()
            
            real_results.append(reals)

    df = pd.concat(real_results, axis=0)
    # df.set_index("date", inplace=True)
    
    fig = make_subplots(rows=3, cols=1)
    
    
    fig.add_trace(
        go.Line(x=df.index, y=df["close"], name="Close", mode="lines"),   
        row=1, col=1
    )
    fig.add_trace(
        go.Figure().add_trace(
            go.Line(x=df.index, y=df["model_vol"], name="Model Volume", mode="lines"),
        ).add_trace(
            go.Line(x=df.index, y=df["log_vol_pct"], name="Real Volume", mode="lines"),
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Figure().add_trace(
            go.Line(x=df.index, y=df["model_amount"], name="Model Volume", mode="lines"),
        ).add_trace(
            go.Line(x=df.index, y=df["amount_pct"], name="Real Volume", mode="lines"),
        ),
    )
    
    fig.show()
    
    pass
            
            


if __name__ == "__main__":
    
    main()