from tqdm import tqdm
import torch

def run_train(
              device, model, loader, criterion, optimizer, scheduler, 
              dashboard_instance, epoch, table, isTrain=True):
    
    title = "Train" if isTrain else "Validation"
    
    batch_loss, batch_rec_loss, batch_kl_loss = 0, 0, 0
    
    if not isTrain:
        result = []
        name = None
        
    for batch_step, (item, reals) in enumerate(tqdm(loader, total=len(loader), desc=f"{title}")):
        item = item.to(device)
        z, z_mu, z_log_var, reconstruction = model(item)
        
        # 손실 계산
        L, rec, kl, likelihood, elbo = criterion(reconstruction, item, model.kl_div)

        # 역전파
        if isTrain: 
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
                
        # 배치 손실 기록
        batch_rec_loss += rec.item()
        batch_kl_loss += kl.item()
        batch_loss += L.item()
        
        dashboard_instance.log({
                f"Batch_{title}": len(loader)*epoch + batch_step,
                f"Batch_{title}/Loss": L.item(),
                f"Batch_{title}/Reconstruction-Error": rec.item(),
                f"Batch_{title}/KL-Div": kl.item(),
                })
        
        batch_step += 1
        
        if not isTrain:
            reals["model_vol"] = reconstruction.squeeze(-1).detach().cpu().numpy()[:, 0].flatten()
            reals["model_amount"] = reconstruction.squeeze(-1).detach().cpu().numpy()[:, 1].flatten()
            reals["loss"] = elbo.detach().cpu().numpy()
            # reals["likelihood"] = likelihood.detach().cpu().numpy().flatten()
            result.append(reals)
            
        
    mt = len(loader)
    
    if isTrain:
        print("[Loss Train]\t\t L: {:.4f}".format(batch_loss/mt))
        if scheduler is not None: scheduler.step()
        return batch_loss / mt, batch_rec_loss / mt, batch_kl_loss / mt
    else:
        import pandas as pd
        print("[Loss Validation]\t\t L: {:.4f}".format(batch_loss/mt))
        
        return batch_loss / mt, batch_rec_loss / mt, batch_kl_loss / mt, pd.concat(result, axis=0)
    