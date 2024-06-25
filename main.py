import torch
import pandas as pd

import os
from torch.utils.data import DataLoader

from utils.training.dataset import eBestDataset
from utils.training.collate import collate

from models.CVAE.cvae import CVAE
from Ztemp.criterion import likelihood_loss

import wandb

from datetime import datetime

from Ztemp.init_dataset import load_dataset
from Ztemp.run_train import run_train

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR

def main(model, criterion, optimizer, scheduler=None, config=None):
    
    epochs = config["epochs"]
    
    with pd.HDFStore("Base.h5", "r") as store:
        keys = store.keys()
        _, test = train_test_split(list(keys), test_size=0.2, shuffle=True, random_state=42)
        train, valid = train_test_split(_, test_size=0.3, shuffle=True, random_state=42)
        
    #     data = load_dataset([keys[8], keys[6]], store, resample=False)
    
    # train_set = eBestDataset(data[0][1], data[0][2], resampled=False)
    # valid_set = eBestDataset(data[1][1], data[1][2], resampled=False)
    
    # train_loader = DataLoader(train_set, batch_size=128, collate_fn=collate, shuffle=True, drop_last=True)
    # valid_loader = DataLoader(valid_set, batch_size=128, collate_fn=collate, shuffle=False, drop_last=True)
    
    
    with wandb.init(reinit=True, project="Finance", name=f"{os.getenv('USERNAME')}-Loss-Optim_{datetime.now().strftime('%Y%m%d %H%M%S')}") as run:
        run.watch(model)

        run.define_metric("Epoch")
        run.define_metric("Train/*", step_metric="Epoch")
        run.define_metric("Validation/*", step_metric="Epoch")
        
        run.define_metric("Batch_Train")
        run.define_metric("Batch_Train/*", step_metric="Batch_Train")
        run.define_metric("Batch_Validation")
        run.define_metric("Batch_Validation/*", step_metric="Batch_Validation")

        valid_table = wandb.Table(columns=["date", "model_vol", "model_amount", "log_vol_pct", "amount_pct"])
        with pd.HDFStore("Base.h5", "r") as store:
            train_data = load_dataset(train, store, resample=False)
        train_set = eBestDataset(pd.concat([subset for key, subset, length in train_data], axis=0), None)
        train_loader = DataLoader(train_set, batch_size=256, collate_fn=collate, shuffle=True, drop_last=True)
        del train_data
        
        for epoch in range(epochs):
            sub_train_loss, sub_train_rec, sub_train_kl = 0, 0, 0
            sub_valid_loss, sub_valid_rec, sub_valid_kl, valid_len = 0, 0, 0, 0
            
            model.train()
            t_loss, t_rec, t_kl_div = run_train(device, model, train_loader, criterion, optimizer, scheduler, run, epoch, None, isTrain=True)
            
            sub_train_loss += t_loss
            sub_train_rec += t_rec
            sub_train_kl += t_kl_div
            
            
            with torch.no_grad():
                model.eval()
                
                with pd.HDFStore("Base.h5", "r") as store:
                    valid_data = load_dataset(valid, store, resample=False)
                for _ in valid_data:
                    key, subset, length = valid_data.pop(0)
                    valid_set = eBestDataset(subset, key)
                    valid_loader = DataLoader(valid_set, batch_size=64, collate_fn=collate, shuffle=False, drop_last=True)
                    v_loss, v_rec, v_kl_div, result_df = run_train(device, model, valid_loader, criterion, optimizer, scheduler, run, epoch, valid_table, isTrain=False)
                    
                    sub_valid_loss += v_loss
                    sub_valid_rec += v_rec
                    sub_valid_kl += v_kl_div
                    valid_len += len(valid_set)
            
                    # result_table = wandb.Table(f"Epoch{epoch}-Result_Table", dataframe=result_df)
            
            run.log({
                "Epoch": epoch,
                "Train/Loss": sub_train_loss / len(train_loader),
                "Train/Reconstruction-Error": sub_train_rec / len(train_loader),
                "Train/KL-Div": sub_train_kl / len(train_loader), 
                
                "Validation/Loss": sub_valid_loss / valid_len,
                "Validation/Reconstruction-Error": sub_valid_rec / valid_len,
                "Validation/KL-Div": sub_valid_kl / valid_len,
                # "Result_Table": result_table
            })

            if epoch % 2 == 0:
                # result_df[["date", "close", "jdiff_vol", "amount", "model_vol", "model_amount", "log_vol_pct", "amount_pct"]].to_csv("Result/result.csv")
                torch.save(model.state_dict(), f"Result/model_{epoch}.pt")
                torch.save(criterion.state_dict(), f"Result/criterion_{epoch}.pt")
            
if __name__ == "__main__":
    
    model = CVAE(latent_dim=10, num_param=2, in_window_size=512, out_window_size=512, scale_flag=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = likelihood_loss(0.001, "MSE")
    scheduler = CosineAnnealingLR(optimizer, T_max=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    criterion.to(device)
    
    config = {
        "epochs": 500
    }
    
    main(model, optimizer=optimizer, criterion=criterion, config=config)
    
    