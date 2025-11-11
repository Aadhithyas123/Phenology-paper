#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Non-episodic TL (vectorized MLP) – internal baseline
"""
Paper-faithful non-episodic TL baseline with robust IO:
- Pretrain on LV / LV* / LV+PT / LV*+PT* (overlap options)
- Reset linear head
- Fine-tune on EE k-shot splits; evaluate on fixed EE test
- Labels are resolved in this priority:
    1) split JSON mapping ("labels" or "pid2label")
    2) NPZ keys (y/label/...)
    3) Parsed from PID (last 10-digit code in filename)

Run:
  python train_baseline.py --config THIS/config_baseline.yaml
"""

import argparse, json, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable
import re

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import yaml
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_yaml(path: str) -> dict:
    with open(path, "r") as f: return yaml.safe_load(f)

def load_json(path: str) -> dict:
    with open(path, "r") as f: return json.load(f)

def load_split_and_labels(split_json: str) -> Tuple[Dict[str, List[str]], Optional[Dict[str,int]]]:
    data = load_json(split_json)
    splits = {k: list(v) for k, v in data.items() if k in ("train","val","test")}
    labels = None
    for key in ("labels","pid2label"):
        if key in data and isinstance(data[key], dict):
            labels = {str(k): int(v) for k,v in data[key].items()}
            break
    return splits, labels

def drop_b10_feature_index(names: List[str]) -> List[int]:
    bad = {"B10","B010","B_10","B 10"}
    return [i for i,n in enumerate(names) if n not in bad]

def resolve_npz_path(root: Path, pid: str) -> Path:
    root = Path(root); p = Path(pid)
    cands = []
    if p.is_absolute() and p.suffix == ".npz":
        cands = [p]
    elif p.suffix == ".npz":
        cands = [root / p.name, root / p.stem]
    else:
        cands = [root / pid, root / f"{pid}.npz"]
    for c in cands:
        if c.is_file(): return c
    raise FileNotFoundError(f"Could not resolve npz for pid='{pid}' under root='{root}'. Tried: {', '.join(str(x) for x in cands)}")

def load_xy_from_npz(path: Path):
    """
    Return (x, y) from NPZ.
    x keys: x/X/features/feat/data/inputs/arr_0 (arr_0 may be a pickled dict)
    y keys: y/Y/label/labels/target/targets/arr_1 (or in that dict)
    """
    z = np.load(path, allow_pickle=True)
    try:
        keys = list(z.files)
        def _from_candidates(cands):
            for k in cands:
                if k in z: return z[k]
            if "arr_0" in z:
                arr0 = z["arr_0"]
                if isinstance(arr0, np.ndarray) and arr0.dtype == object and arr0.shape == ():
                    obj = arr0.item()
                    if isinstance(obj, dict):
                        for k in cands:
                            if k in obj: return obj[k]
            return None
        x = _from_candidates(["x","X","features","feat","data","inputs","arr_0"])
        y = _from_candidates(["y","Y","label","labels","target","targets","arr_1"])
        if x is None:
            raise KeyError(f"No feature key found in {path}. Available: {keys}")
        return x, y
    finally:
        z.close()

_pid_code_re = re.compile(r"(\d{10})(?=\.npz$|$)")
def extract_label_code_from_pid(pid: str) -> Optional[str]:
    """
    Extract the 10-digit crop code from PID/filename, e.g.,
    'LV005_12418831_3301030000.npz' -> '3301030000'
    """
    name = Path(pid).name
    m = _pid_code_re.search(name)
    if m: return m.group(1)
    # try last underscore token w/o extension
    stem = Path(pid).stem
    tok = stem.split("_")[-1]
    if tok.isdigit() and len(tok) >= 4:  # fallback
        return tok
    return None

def build_code_index(split_paths: Iterable[str]) -> Dict[str,int]:
    """
    Scan all split JSONs and collect label codes from PIDs, build consistent code->index mapping.
    """
    codes = []
    for sp in split_paths:
        if not sp: continue
        splits, labels = load_split_and_labels(sp)
        for subset in ("train","val","test"):
            for pid in splits.get(subset, []):
                if labels and str(pid) in labels:  # mapping exists; skip
                    continue
                code = extract_label_code_from_pid(str(pid))
                if code is not None: codes.append(code)
    unique = sorted(set(codes))
    return {c:i for i,c in enumerate(unique)}

def code2idx_label_for_pid(pid: str, code2idx: Optional[Dict[str,int]]) -> Optional[int]:
    if code2idx is None: return None
    code = extract_label_code_from_pid(pid)
    if code is None: return None
    return code2idx.get(code, None)

def ensure_scalar_label(y):
    yarr = np.array(y)
    if yarr.shape == (): return int(yarr.item())
    yarr = yarr.squeeze()
    return int(yarr[()]) if yarr.shape == () else int(yarr)

def clip_idx(idx: List[int], C: int) -> List[int]:
    return [i for i in idx if i < C]

# ----------------------------
# Dataset (fallback; swap with your real dataset if needed)
# ----------------------------
class MinimalDataset(Dataset):
    def __init__(self,
                 root_npz: str,
                 split_json: str,
                 subset: str,
                 normalize: str = "zscore",
                 per_parcel: bool = True,
                 drop_b10: bool = True,
                 feature_names: Optional[List[str]] = None,
                 only_pids: Optional[List[str]] = None,
                 labels_by_pid: Optional[Dict[str,int]] = None,
                 code2idx: Optional[Dict[str,int]] = None):
        super().__init__()
        self.root = Path(root_npz)
        self.splits, json_labels = load_split_and_labels(split_json)
        self.pids = list(only_pids) if only_pids is not None else list(self.splits[subset])
        self.normalize = normalize
        self.per_parcel = per_parcel
        self.drop_b10 = drop_b10
        self.feature_names = feature_names or ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]
        self.keep_idx = drop_b10_feature_index(self.feature_names) if drop_b10 else None
        self.labels_by_pid = labels_by_pid if labels_by_pid is not None else json_labels
        self.code2idx = code2idx

        self.mean, self.std = None, None
        if normalize == "zscore" and not per_parcel and len(self.pids) > 0:
            samp = self.pids[:min(2000, len(self.pids))]
            xs = []
            for pid in samp:
                fpath = resolve_npz_path(self.root, pid)
                x, _ = load_xy_from_npz(fpath)
                x = self._to_vec(x)
                xs.append(x)
            allx = np.stack(xs, axis=0)  # (N,C)
            self.mean = allx.mean(axis=0, keepdims=True)
            self.std  = allx.std(axis=0, keepdims=True) + 1e-6

    def _to_vec(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:  # (T,C)
            if self.keep_idx is not None: x = x[:, clip_idx(self.keep_idx, x.shape[1])]
            x = x.mean(axis=0)
        elif x.ndim == 1:
            if self.keep_idx is not None: x = x[clip_idx(self.keep_idx, x.shape[0])]
        else:
            raise RuntimeError(f"Unexpected feature shape {x.shape}")
        return x.astype(np.float32)

    def __len__(self): return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        fpath = resolve_npz_path(self.root, pid)
        x, y_file = load_xy_from_npz(fpath)
        x = self._to_vec(x)

        # Label priority: labels_by_pid → NPZ → code from PID
        if self.labels_by_pid is not None and str(pid) in self.labels_by_pid:
            y = int(self.labels_by_pid[str(pid)])
        elif y_file is not None:
            y = ensure_scalar_label(y_file)
        else:
            y = code2idx_label_for_pid(str(pid), self.code2idx)
            if y is None:
                raise RuntimeError(f"No label for pid={pid}: provide mapping in split JSON or ensure PID contains code.")

        if self.normalize == "zscore":
            if self.per_parcel:
                mu, sd = x.mean(keepdims=True), x.std(keepdims=True) + 1e-6
                x = (x - mu) / sd
            else:
                x = (x - self.mean.squeeze()) / self.std.squeeze()

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

# ----------------------------
# Model (simple backbone + linear head)
# ----------------------------
class MLPBackbone(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 256, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        layers, dim = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, d_model), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            dim = d_model
        self.net = nn.Sequential(*layers)
        self.out_dim = d_model
    def forward(self, x): return self.net(x)

class LinearHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__(); self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, z): return self.fc(z)

class TLModel(nn.Module):
    def __init__(self, in_dim: int, d_model: int, n_layers: int, dropout: float, n_classes: int):
        super().__init__()
        self.backbone = MLPBackbone(in_dim, d_model, n_layers, dropout)
        self.head = LinearHead(self.backbone.out_dim, n_classes)
    def reset_head(self, n_classes: int):
        self.head = LinearHead(self.backbone.out_dim, n_classes)
    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)

# ----------------------------
# Train/Eval helpers
# ----------------------------
def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=True, drop_last=False)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, metrics=("acc",)) -> Dict[str,float]:
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(dim=1)
        ys.append(yb.cpu().numpy()); ps.append(pred.cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    out = {}
    if "acc" in metrics:     out["acc"] = float(accuracy_score(y_true, y_pred))
    if "macro_f1" in metrics:out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    if "kappa" in metrics:   out["kappa"] = float(cohen_kappa_score(y_true, y_pred))
    return out

def early_stop_update(best_metric, current_metric, patience_cnt, mode="max"):
    improve = (current_metric > best_metric) if mode=="max" else (current_metric < best_metric)
    if improve: return current_metric, 0, True
    return best_metric, patience_cnt+1, False

def downsample_class_to_median(pids: List[str], labels_by_pid: Dict[str,int], class_id: int) -> List[str]:
    from collections import defaultdict
    per_class = defaultdict(list)
    for pid in pids:
        per_class[labels_by_pid[str(pid)]].append(pid)
    sizes = [len(v) for c,v in per_class.items() if c != class_id]
    if not sizes: return pids
    target = int(np.median(sizes))
    keep = []
    for c,plist in per_class.items():
        if c == class_id and len(plist) > target:
            keep += random.sample(plist, target)
        else:
            keep += plist
    return keep

def infer_in_dim(root_npz: str, split_json: str, drop_b10: bool, feat_names: List[str]) -> int:
    splits, _ = load_split_and_labels(split_json)
    probe_pid = splits["train"][0]
    probe_path = resolve_npz_path(Path(root_npz), probe_pid)
    x_probe, _ = load_xy_from_npz(probe_path)
    if x_probe.ndim == 2: C_full = x_probe.shape[1]
    elif x_probe.ndim == 1: C_full = x_probe.shape[0]
    else: raise RuntimeError(f"Unexpected feature shape {x_probe.shape} in {probe_path}")
    keep_idx = drop_b10_feature_index(feat_names) if drop_b10 else list(range(C_full))
    keep_idx = clip_idx(keep_idx, C_full)
    return len(keep_idx)

def infer_num_classes_from_ds(ds: "MinimalDataset", root_npz: str) -> int:
    # Prefer JSON mapping if present
    if ds.labels_by_pid is not None:
        lbls = [ds.labels_by_pid[str(pid)] for pid in ds.pids if str(pid) in ds.labels_by_pid]
        if len(lbls) > 0: return int(np.max(lbls) + 1)
    # Else use code2idx (robust for this project)
    if ds.code2idx is not None:
        # Collect only labels present in this dataset
        ys = []
        for pid in ds.pids:
            y = code2idx_label_for_pid(str(pid), ds.code2idx)
            if y is not None: ys.append(y)
        if len(ys) > 0: return int(np.max(ys) + 1)
    # Fallback to NPZ labels if any
    lbls = []
    for pid in ds.pids[:5000]:
        fpath = resolve_npz_path(Path(root_npz), pid)
        _, y = load_xy_from_npz(fpath)
        if y is not None: lbls.append(ensure_scalar_label(y))
    if len(lbls) == 0:
        raise RuntimeError("Could not infer number of classes: no labels in split JSON, NPZ, or PID codes.")
    return int(np.max(lbls) + 1)

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--subset-train", type=str, default="train")
    ap.add_argument("--subset-val", type=str, default="val")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(cfg.get("device", "cuda") if args.device is None else args.device)

    seeds: List[int] = cfg["eval"].get("seeds", [42])
    ds_cfg = cfg["dataset"]
    root_npz = ds_cfg["root_npz"]
    drop_b10 = bool(ds_cfg.get("drop_b10", True))
    normalize = ds_cfg.get("normalize", "zscore")

    enc_cfg = cfg.get("encoder", {})
    d_model = int(enc_cfg.get("dim", 256))
    depth = int(enc_cfg.get("depth", 2))
    dropout = float(enc_cfg.get("dropout", 0.1))

    pt_cfg = cfg["pretraining"]
    scenario = pt_cfg.get("scenario", "LV")
    pretrain_json_map = {
        "LV": ds_cfg["pretrain_json"],
        "LV_overlap": ds_cfg.get("pretrain_json_overlap", ds_cfg["pretrain_json"]),
        "LVPT": ds_cfg.get("pretrain_json_lvpt", ds_cfg["pretrain_json"]),
        "LVPT_overlap": ds_cfg.get("pretrain_json_lvpt_overlap", ds_cfg["pretrain_json"]),
    }
    pretrain_json = pretrain_json_map[scenario]
    pt_bs = int(pt_cfg.get("batch_size", 128))
    pt_epochs = int(pt_cfg.get("epochs", 150))
    pt_patience = int(pt_cfg.get("early_stopping_patience", 15))
    pt_lr = float(pt_cfg.get("lr", 3e-4))
    pt_cos = pt_cfg.get("cosine_anneal", {"enabled": True, "cycles": 1})
    do_downsample = bool(pt_cfg.get("downsample_meadow_to_median", True))
    MEADOW_ID = pt_cfg.get("meadow_label_id", None)  # optional known ID

    ft_cfg = cfg["finetune"]
    k_grid: List[int] = list(ft_cfg.get("k_grid", [1,5,10,20,100,200,500]))
    ft_bs = int(ft_cfg.get("batch_size", 16))
    ft_epochs = int(ft_cfg.get("epochs", 200))
    ft_patience = int(ft_cfg.get("early_stopping_patience", 5))
    lr_mode = ft_cfg.get("lr_mode", "separate")
    lr_all = float(ft_cfg.get("lr", {}).get("same", 3e-4))
    lr_head_only = float(ft_cfg.get("lr", {}).get("head_only", 3e-4))
    lr_sep_head = float(ft_cfg.get("lr", {}).get("separate", {}).get("head", 3e-4))
    lr_sep_back = float(ft_cfg.get("lr", {}).get("separate", {}).get("backbone", 1e-4))

    metrics = cfg["eval"].get("metrics", ["acc"])
    outdir = Path(cfg["paths"]["outputs_dir"]); outdir.mkdir(parents=True, exist_ok=True)

    # ---- Build a global code->index map so labels are consistent across splits
    split_paths = [pretrain_json] + [ds_cfg["finetune_json_tpl"].format(k=k) for k in k_grid]
    code2idx = build_code_index(split_paths)  # empty is fine; then fallbacks will be used

    # ---- Infer input dim
    feat_names = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]
    in_dim = infer_in_dim(root_npz, pretrain_json, drop_b10, feat_names)

    all_results = []
    for seed in seeds:
        set_seed(seed)
        seed_dir = outdir / f"seed_{seed}"; seed_dir.mkdir(parents=True, exist_ok=True)

        # Pretraining datasets
        splits_pt, labels_pt = load_split_and_labels(pretrain_json)
        train_pids_pt = list(splits_pt["train"])
        if do_downsample and (labels_pt is not None) and (MEADOW_ID is not None):
            train_pids_pt = downsample_class_to_median(train_pids_pt, labels_pt, int(MEADOW_ID))

        train_ds_pt = MinimalDataset(root_npz, pretrain_json, subset="train",
                                     normalize=normalize, per_parcel=True,
                                     drop_b10=drop_b10, feature_names=feat_names,
                                     only_pids=train_pids_pt, labels_by_pid=labels_pt, code2idx=code2idx)
        val_ds_pt = MinimalDataset(root_npz, pretrain_json, subset="val",
                                   normalize=normalize, per_parcel=True,
                                   drop_b10=drop_b10, feature_names=feat_names,
                                   labels_by_pid=labels_pt, code2idx=code2idx)

        train_ld_pt = make_loader(train_ds_pt, pt_bs, True,  cfg.get("num_workers", 8))
        val_ld_pt   = make_loader(val_ds_pt,   pt_bs, False, cfg.get("num_workers", 8))

        # Model
        n_classes_pre = infer_num_classes_from_ds(val_ds_pt, root_npz)
        model = TLModel(in_dim=in_dim, d_model=d_model, n_layers=depth, dropout=dropout,
                        n_classes=n_classes_pre).to(device)

        # Pretrain
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=pt_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, pt_epochs // int(pt_cos.get("cycles", 1)))) if pt_cos.get("enabled", True) else None

        best_val, best_state, patience = -1.0, None, 0
        for epoch in range(1, pt_epochs+1):
            model.train()
            for xb, yb in train_ld_pt:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); opt.step()
            if scheduler is not None: scheduler.step()
            acc = evaluate(model, val_ld_pt, device, metrics=("acc",))["acc"]
            best_val, patience, improved = early_stop_update(best_val, acc, patience, "max")
            if improved: best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            if patience >= pt_patience: break

        if best_state is not None:
            model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
        torch.save(model.backbone.state_dict(), seed_dir / "pretrained_backbone.pt")

        # Fine-tune over k
        results_seed = {"seed": seed, "per_k": {}}
        for k in k_grid:
            ft_json = ds_cfg["finetune_json_tpl"].format(k=k)
            splits_ft, labels_ft = load_split_and_labels(ft_json)

            train_ds_ft = MinimalDataset(root_npz, ft_json, subset="train",
                                         normalize=normalize, per_parcel=True,
                                         drop_b10=drop_b10, feature_names=feat_names,
                                         labels_by_pid=labels_ft, code2idx=code2idx)
            val_ds_ft   = MinimalDataset(root_npz, ft_json, subset="val",
                                         normalize=normalize, per_parcel=True,
                                         drop_b10=drop_b10, feature_names=feat_names,
                                         labels_by_pid=labels_ft, code2idx=code2idx)
            test_ds_ft  = MinimalDataset(root_npz, ft_json, subset="test",
                                         normalize=normalize, per_parcel=True,
                                         drop_b10=drop_b10, feature_names=feat_names,
                                         labels_by_pid=labels_ft, code2idx=code2idx)

            train_ld_ft = make_loader(train_ds_ft, ft_bs, True,  cfg.get("num_workers", 8))
            val_ld_ft   = make_loader(val_ds_ft,   ft_bs, False, cfg.get("num_workers", 8))
            test_ld_ft  = make_loader(test_ds_ft,  ft_bs, False, cfg.get("num_workers", 8))

            # Reset head to match current class set
            n_classes_ee = infer_num_classes_from_ds(test_ds_ft, root_npz)
            model.reset_head(n_classes_ee); model.to(device)

            if lr_mode == "head_only":
                for p in model.backbone.parameters(): p.requires_grad = False
                opt = torch.optim.Adam([{"params": model.head.parameters(), "lr": lr_head_only}])
            elif lr_mode == "same":
                for p in model.backbone.parameters(): p.requires_grad = True
                opt = torch.optim.Adam([{"params": model.parameters(), "lr": lr_all}])
            else:
                for p in model.backbone.parameters(): p.requires_grad = True
                opt = torch.optim.Adam([
                    {"params": model.backbone.parameters(), "lr": lr_sep_back},
                    {"params": model.head.parameters(), "lr": lr_sep_head},
                ])

            criterion = nn.CrossEntropyLoss()
            scheduler = None
            best_val, best_state, patience = -1.0, None, 0
            for epoch in range(1, ft_epochs+1):
                model.train()
                for xb, yb in train_ld_ft:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); opt.step()
                if scheduler is not None: scheduler.step()
                acc = evaluate(model, val_ld_ft, device, metrics=("acc",))["acc"]
                best_val, patience, improved = early_stop_update(best_val, acc, patience, "max")
                if improved: best_state = {k:v.cpu() for k,v in model.state_dict().items()}
                if patience >= ft_patience: break

            if best_state is not None:
                model.load_state_dict({k:v.to(device) for k,v in best_state.items()})

            test_metrics = evaluate(model, test_ld_ft, device, metrics=metrics)
            results_seed["per_k"][k] = test_metrics
            with open(seed_dir / f"results_k{k}.json", "w") as f:
                json.dump(test_metrics, f, indent=2)

        all_results.append(results_seed)

    # Aggregate
    # Aggregate across seeds
    agg = {}
    ks = sorted(k_grid)
    for k in ks:
        bucket = {m: [] for m in metrics}
        for r in all_results:
            tm = r["per_k"][k]
            for m in metrics:
                bucket[m].append(tm[m])

        agg[k] = {}
        for m, vals in bucket.items():
            agg[k][f"{m}_mean"] = float(np.mean(vals))
            agg[k][f"{m}_std"]  = float(np.std(vals))

    with open(outdir / "summary.json", "w") as f:
        json.dump({"seeds": seeds, "k_grid": ks, "aggregate": agg, "per_seed": all_results}, f, indent=2)

    print("\n=== Aggregate (mean ± std over seeds) ===")
    for k in ks:
        row = [f"k={k}"]
        for m in metrics:
            row.append(f"{m}={agg[k][f'{m}_mean']:.3f}±{agg[k][f'{m}_std']:.3f}")
        print("  " + "  ".join(row))


if __name__ == "__main__":
    main()
