import torch
import torch.nn as nn
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import math
import struct
from io import StringIO
from datetime import date
import torch.nn.functional as F
import os
import pickle
import plotly.graph_objects as go

from torch.utils.data import DataLoader, Dataset
from ds import DS, collate

import plotly.graph_objects as go
import plotly.express as px

from models.CVAE.cvae import CVAE
from Ztemp.criterion import likelihood_loss

import wandb
from datetime import datetime
from tqdm import tqdm

from sklearn.preprocessing import Normalizer
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash


def get_dataset(keys, store, resample=True):
    dataset = [] # name, subset
    
    for key in tqdm(keys, total=len(keys), desc="Dataset"):
        sub_set = store.get(key)
        
        subset = [] # 1s_item
        length = []
        
        sub_set["date"] = sub_set["date"].astype(str)
        sub_set["amount"] = sub_set["close"] * sub_set["jdiff_vol"]
        sub_set["close_pct"] = sub_set["close"].pct_change()
        sub_set["vol_pct"] = sub_set["jdiff_vol"].pct_change().fillna(0)
        
        #########
        sub_set["amount_pct"] = np.log(sub_set["amount"]).pct_change().fillna(0)
        sub_set["log_vol_pct"] = np.log((sub_set["jdiff_vol"] / sub_set["jdiff_vol"].shift(1))).fillna(0) * 0.1 
        #########
        
        sub_set["log_amount_pct"] = np.log((sub_set["amount"] / sub_set["amount"].shift(1))).fillna(0)
        sub_set["log_close_pct"] = (100*np.log((sub_set["close"] / sub_set["close"].shift(1))).fillna(0))
        sub_set["date"] = pd.to_datetime(sub_set["date"] + " " + sub_set["time"], format="%Y%m%d %H%M%S")
        
        if resample:
            [[subset.append(item), length.append(len(item))] for _, item in tqdm(sub_set.resample("1s", on="date"), total=len(sub_set.resample("1s", on="date")), desc="subset") if not item.empty]
            dataset.append((key, subset, length))
            
        else:
            dataset.append((key, sub_set, len(sub_set)))
        
    return dataset

with pd.HDFStore("Base.h5", "r") as store:
    keys = store.keys()
    data = get_dataset([keys[6], keys[8]], store, resample=False)
    
model = CVAE(latent_dim=10, num_param=2, in_window_size=512, out_window_size=512, scale_flag=1)

dataset = DS(data[0][1], data[0][2], False)
val_dataset = DS(data[1][1], data[1][2], False)

loader = DataLoader(dataset, batch_size=32, collate_fn=collate, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate, shuffle=False, drop_last=True)

epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = likelihood_loss(0.001, "MSE")

isTrain = True

model.to(device)

# 실시간 시각화 분리하기
# app = Dash(__name__)
# app.layout = html.Div([
#     dcc.Graph(id='real-time-plot'),
#     dcc.Interval(
#         id='interval-component',
#         interval=10*1000,  # in milliseconds
#         n_intervals=0
#     )
# ])
# # Define callback to update the plot in real-time
# @app.callback(Output('real-time-plot', 'figure'),
#               [Input('interval-component', 'n_intervals')])

# def update_graph(new_data):
#     # Load new data from Excel file
    
#     # Update the plot
#     fig = px.line(new_data, x="date", y=["log_vol_pct", "amount_pct", "model_vol", "model_amount"], title='Real-time Data Plot')
    
#     return fig

with wandb.init(reinit=True, project="Finance", name=f"{os.getenv('USERNAME')}-Loss-Optim_{datetime.now().strftime('%Y%m%d %H%M%S')}") as run:
    
    run.define_metric("Metric/Epoch")
    run.define_metric("Metric/*", step_metric="Metric/Epoch")
    
    run.define_metric("Train/step")
    run.define_metric("Train/*", step_metric="Train/step")
    run.define_metric("Validation/step")
    run.define_metric("Validation/*", step_metric="Validation/step")
    
    # run.define_metric("Predict/Train_step")
    # run.define_metric("Predict/Train/*", step_metric="Predict/Train_step")
    # run.define_metric("Predict/Validation_step")
    # run.define_metric("Predict/Validation/*", step_metric="Predict/Validation_step")
    
    results = [[]] * epochs
    
    val_result = []
    
    for epoch in range(1, epochs+1):
        
        _step, _val_step = 0, 0
        
        _rec_loss, _kl_loss, _loss = 0, 0, 0
        _val_rec_loss, _val_kl_loss, _val_loss = 0, 0, 0
        
        model.train()
        for item, reals in loader:
            item = item.to(device)
            
            trg = model(item)
            L, rec, kl, likelihood = criterion(trg, item, model.kl_div)
            
            optimizer.zero_grad()
            L.backward()
            optimizer.step()

            _rec_loss += rec.item()
            _kl_loss += kl.item()
            _loss += L.item()
            _step += 1
            
            run.log({                
                f"Train/step": len(loader)*epoch + _step,
                f"Train/Loss": L.item(),
                f"Train/Reconstruction-Error": rec.item(),
                f"Train/KL-Div": kl.item()
            })
        print("[Loss Train]\t\t L: {:.4f}".format(_loss/len(loader)))
        
        with torch.no_grad():
            model.eval()
            for item, reals in val_loader:
                item = item.to(device)
                
                trg = model(item)
                
                L, rec, kl, likelihood = criterion(trg, item, model.kl_div)
            
                _val_rec_loss += rec.item()
                _val_kl_loss += kl.item()
                _val_loss += L.item()
                _val_step += 1
                
                run.log({                
                    f"Validation/step": len(val_loader)*epoch + _val_step,
                    f"Validation/Loss": L.item(),
                    f"Validation/Reconstruction-Error": rec.item(),
                    f"Validation/KL-Div": kl.item()
                })
                
                reals["model_vol"] = trg.squeeze(-1).detach().cpu().numpy()[:, 0].flatten()
                reals["model_amount"] = trg.squeeze(-1).detach().cpu().numpy()[:, 1].flatten()
                val_result.append(reals)

        print("[Loss Validation]\t\t L: {:.4f}".format(_val_loss/len(val_loader)))
        
        run.log({
            "Metric/Epoch": epoch,
            "Metric/Train/Loss": _loss/len(loader),
            "Metric/Train/Reconstruction-Error": _rec_loss/len(loader),
            "Metric/Train/KL-Div": _kl_loss/len(loader),
            
            "Metric/Validation/Loss": _val_loss/len(val_loader),
            "Metric/Validation/Reconstruction-Error": _val_rec_loss/len(val_loader),
            "Metric/Validation/KL-Div": _val_kl_loss/len(val_loader),
        })

    pd.concat([pd.DataFrame(trg.squeeze(-1).detach().cpu().numpy(), columns=["vol", "amount"]), pd.DataFrame(item.squeeze(-1).detach().cpu().numpy(), columns=["model_vol", "model_amount"])], axis=1).plot()
    pass
# pd.concat([pd.DataFrame(trg.squeeze(-1).detach().cpu().numpy(), columns=["vol", "amount"]), pd.DataFrame(item.squeeze(-1).detach().cpu().numpy(), columns=["model_vol", "model_amount"])], axis=1).plot()