import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from collections import defaultdict

@torch.no_grad()
def _split_support_query(batch: Dict[str, torch.Tensor],
                         n_way: int, k_shot: int, q_query: int,
                         device: torch.device):
    """
    Assumes sampler yielded contiguous blocks per class: for each class c,
    first k_shot are support, next q_query are query.
    """
    x, mask, y = batch["x"].to(device), batch["mask"].to(device), batch["y"].to(device)
    B = x.size(0)
    ex_per_class = k_shot + q_query
    assert B == (n_way * ex_per_class), f"batch size {B} != n_way*(k+q)"
    # reorder indices by class-blocks
    xs, ms, ys, xq, mq, yq = [], [], [], [], [], []
    for i in range(n_way):
        s = i * ex_per_class
        e = s + ex_per_class
        xb, mb, yb = x[s:e], mask[s:e], y[s:e]
        xs.append(xb[:k_shot]);   ms.append(mb[:k_shot]);   ys.append(yb[:k_shot])
        xq.append(xb[k_shot:]);   mq.append(mb[k_shot:]);   yq.append(yb[k_shot:])
    xs = torch.cat(xs, dim=0)  # [n_way*k, T, C]
    ms = torch.cat(ms, dim=0)
    ys = torch.cat(ys, dim=0)
    xq = torch.cat(xq, dim=0)  # [n_way*q, T, C]
    mq = torch.cat(mq, dim=0)
    yq = torch.cat(yq, dim=0)
    return {"xs": xs, "ms": ms, "ys": ys, "xq": xq, "mq": mq, "yq": yq}

def run_episode(model_enc, head, batch, n_way, k_shot, q_query, device):
    parts = _split_support_query(batch, n_way, k_shot, q_query, device)
    # encode
    _, z_s = model_enc(parts["xs"], parts["ms"])  # [n_way*k, D]
    _, z_q = model_enc(parts["xq"], parts["mq"])  # [n_way*q, D]
    logits, classes = head(z_s, parts["ys"], z_q) # [n_way*q, n_way]
    # remap yq into 0..n_way-1 over this episode
    y_map = {int(c.item()): i for i, c in enumerate(classes)}
    yq_local = torch.tensor([y_map[int(t)] for t in parts["yq"].tolist()], device=device)
    loss = F.cross_entropy(logits, yq_local)
    pred = logits.argmax(dim=1)
    acc = (pred == yq_local).float().mean()
    return loss, acc

def train_one_epoch(model_enc, head, loader, optim, scheduler, device, n_way, k_shot, q_query, log):
    model_enc.train(); head.train()
    losses, accs = [], []
    for it, batch in enumerate(loader):
        optim.zero_grad()
        loss, acc = run_episode(model_enc, head, batch, n_way, k_shot, q_query, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_enc.parameters(), 1.0)
        optim.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item()); accs.append(float(acc))
        if (it+1) % 20 == 0:
            log.info(f"[train] it {it+1}/{len(loader)}  loss={loss.item():.4f}  acc={float(acc):.3f}")
    return sum(losses)/max(1,len(losses)), sum(accs)/max(1,len(accs))

@torch.no_grad()
def evaluate(model_enc, head, loader, device, n_way, k_shot, q_query, log):
    model_enc.eval(); head.eval()
    losses, accs = [], []
    for it, batch in enumerate(loader):
        loss, acc = run_episode(model_enc, head, batch, n_way, k_shot, q_query, device)
        losses.append(loss.item()); accs.append(float(acc))
    mloss = sum(losses)/max(1,len(losses)); macc = sum(accs)/max(1,len(accs))
    log.info(f"[val] episodes={len(loader)}  loss={mloss:.4f}  acc={macc:.3f}")
    return mloss, macc
