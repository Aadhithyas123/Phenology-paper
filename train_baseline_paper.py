#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_paperlike.py — Paper-faithful non-episodic TL baseline (strict k-shot)
Key points:
- Pretraining is performed PER-k with LV* filtered by EE (train+val) classes for that k.
- Sequence encoder (tiny Transformer with CLS).
- FT freezes backbone for small k (k <= 20) to avoid inflated low-k.
- FT evaluates the LAST epoch (no best-val restore).
- Strong leakage checks + exact k-per-class assertions.
"""

import argparse, json, random, re, csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Set
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import yaml
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

# ----------------- utils -----------------
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

# PID code: last 10-digit token (e.g., LV005_..._3301030000.npz -> 3301030000)
_pid_code_re = re.compile(r"(\d{10})(?=\.npz$|$)")
def extract_label_code_from_pid(pid: str) -> Optional[str]:
    name = Path(pid).name
    m = _pid_code_re.search(name)
    if m: return m.group(1)
    stem = Path(pid).stem
    tok = stem.split("_")[-1]
    if tok.isdigit() and len(tok) >= 4:
        return tok
    return None

def ensure_scalar_label(y):
    yarr = np.array(y)
    if yarr.shape == (): return int(yarr.item())
    yarr = yarr.squeeze()
    return int(yarr[()]) if yarr.shape == () else int(yarr)

def _codes_from_pids(pids: Iterable[str]) -> Set[str]:
    out = set()
    for pid in pids:
        c = extract_label_code_from_pid(str(pid))
        if c is not None: out.add(c)
    return out

def assert_disjoint(a: Iterable[str], b: Iterable[str], namea: str, nameb: str):
    A, B = set(map(str,a)), set(map(str,b))
    inter = A & B
    if len(inter) > 0:
        raise AssertionError(f"Leakage between {namea} and {nameb}: {len(inter)} shared PIDs (e.g., {list(sorted(inter))[:5]})")

# build code->index **without using test** (train/val only)
def build_code_index_no_test(split_paths: Iterable[str]) -> Dict[str,int]:
    codes = set()
    for sp in split_paths:
        if not sp: continue
        splits, labels = load_split_and_labels(sp)
        for subset in ("train","val"):
            for pid in splits.get(subset, []):
                if labels and str(pid) in labels:
                    continue
                c = extract_label_code_from_pid(str(pid))
                if c is not None: codes.add(c)
    unique = sorted(codes)
    return {c:i for i,c in enumerate(unique)}

def code2idx_label_for_pid(pid: str, code2idx: Optional[Dict[str,int]]) -> Optional[int]:
    if code2idx is None: return None
    code = extract_label_code_from_pid(pid)
    if code is None: return None
    return code2idx.get(code, None)

# ----------------- dataset (sequence) -----------------
class SeqDataset(Dataset):
    """Returns (x, y, T) where x is (T,C'), z-scored per parcel over time."""
    def __init__(self,
                 root_npz: str,
                 split_json: str,
                 subset: str,
                 drop_b10: bool = True,
                 normalize: str = "zscore",
                 only_pids: Optional[List[str]] = None,
                 labels_by_pid: Optional[Dict[str,int]] = None,
                 code2idx: Optional[Dict[str,int]] = None):
        super().__init__()
        self.root = Path(root_npz)
        self.splits, json_labels = load_split_and_labels(split_json)
        self.pids = list(only_pids) if only_pids is not None else list(self.splits[subset])
        self.drop_b10 = drop_b10
        self.normalize = normalize
        self.labels_by_pid = labels_by_pid if labels_by_pid is not None else json_labels
        self.code2idx = code2idx

    def __len__(self): return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        fpath = resolve_npz_path(self.root, pid)
        x, yf = load_xy_from_npz(fpath)  # x: (T,C) or (C,)
        x = self._to_TxC(x)

        if self.normalize == "zscore":
            mu = x.mean(axis=0, keepdims=True)
            sd = x.std(axis=0, keepdims=True) + 1e-6
            x = (x - mu) / sd

        if self.labels_by_pid is not None and str(pid) in self.labels_by_pid:
            y = int(self.labels_by_pid[str(pid)])
        elif yf is not None:
            y = ensure_scalar_label(yf)
        else:
            y = code2idx_label_for_pid(str(pid), self.code2idx)
            if y is None:
                raise RuntimeError(f"No label for pid={pid}. Provide mapping or ensure code in PID.")
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(y, dtype=torch.long), x.shape[0]

    def _to_TxC(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            C_full = x.shape[1]
            if self.drop_b10 and C_full >= 12:
                if C_full == 13:
                    keep = [0,1,2,3,4,5,6,7,8,9,11,12]  # drop channel 10 (0-based)
                    x = x[:, keep]
            return x
        elif x.ndim == 1:
            if self.drop_b10 and x.shape[0] == 13:
                keep = [0,1,2,3,4,5,6,7,8,9,11,12]
                x = x[keep]
            return x[None,:]
        else:
            raise RuntimeError(f"Unexpected feature shape {x.shape}")

def collate_seq(batch):
    xs, ys, Ts = zip(*batch)
    C = xs[0].shape[1]
    T_max = max(Ts)
    Xp = torch.zeros(len(xs), T_max, C, dtype=xs[0].dtype)
    mask = torch.ones(len(xs), T_max, dtype=torch.bool)  # True=PAD
    for i,(x,t) in enumerate(zip(xs, Ts)):
        Xp[i, :t, :] = x
        mask[i, :t] = False
    y = torch.stack(ys, 0)
    return Xp, y, mask

# ----------------- model -----------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x):  # (B,T,D)
        return x + self.pe[:x.size(1)].unsqueeze(0)

class TinyTSFormer(nn.Module):
    def __init__(self, in_channels: int, n_classes: int,
                 d_model: int=256, nhead: int=8, nlayers: int=2,
                 ff_mult: int=4, dropout: float=0.1, max_len: int=1024):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.cls = nn.Parameter(torch.zeros(1,1,d_model))
        self.pe = SinusoidalPositionalEncoding(d_model, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=ff_mult*d_model, dropout=dropout,
            batch_first=True, norm_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, n_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x, key_padding_mask=None):
        z = self.proj(x)                 # (B,T,D)
        z = self.pe(z)
        B = z.size(0)
        cls = self.cls.expand(B, -1, -1) # (B,1,D)
        z = torch.cat([cls, z], dim=1)   # (B,1+T,D)
        if key_padding_mask is not None:
            pad = torch.zeros(B,1, dtype=key_padding_mask.dtype, device=key_padding_mask.device)
            mask = torch.cat([pad, key_padding_mask], dim=1)  # (B,1+T)
        else:
            mask = None
        h = self.encoder(z, src_key_padding_mask=mask)
        cls_out = h[:,0,:]
        return self.head(cls_out)

# ----------------- helpers -----------------
def infer_in_dim(root_npz: str, split_json: str, drop_b10: bool) -> int:
    splits, _ = load_split_and_labels(split_json)
    probe_pid = splits["train"][0]
    x, _ = load_xy_from_npz(resolve_npz_path(Path(root_npz), probe_pid))
    if x.ndim == 2:
        C = x.shape[1]
    elif x.ndim == 1:
        C = x.shape[0]
    else:
        raise RuntimeError(f"Unexpected x shape {x.shape}")
    if drop_b10 and C == 13:
        return 12
    return C

def infer_num_classes_from_pids(pids: List[str],
                                labels_by_pid: Optional[Dict[str,int]],
                                root_npz: str,
                                code2idx: Optional[Dict[str,int]]) -> int:
    if labels_by_pid:
        ys = [labels_by_pid[str(pid)] for pid in pids if str(pid) in labels_by_pid]
        if len(ys)>0: return int(np.max(ys)+1)
    if code2idx:
        ys = []
        for pid in pids:
            y = code2idx_label_for_pid(str(pid), code2idx)
            if y is not None: ys.append(y)
        if len(ys)>0: return int(np.max(ys)+1)
    ys = []
    for pid in pids[:5000]:
        _, y = load_xy_from_npz(resolve_npz_path(Path(root_npz), pid))
        if y is not None: ys.append(ensure_scalar_label(y))
    if len(ys)==0:
        raise RuntimeError("Unable to infer class count; provide labels in JSON or ensure PID codes.")
    return int(np.max(ys)+1)

def downsample_class_to_median(pids: List[str], labels_by_pid: Dict[str,int], class_id: int) -> List[str]:
    per = {}
    for pid in pids:
        per.setdefault(labels_by_pid[str(pid)], []).append(pid)
    sizes = [len(v) for c,v in per.items() if c != class_id]
    if not sizes: return pids
    target = int(np.median(sizes))
    keep = []
    for c, plist in per.items():
        if c == class_id and len(plist) > target:
            keep += random.sample(plist, target)
        else:
            keep += plist
    return keep

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, metrics=("acc",)):
    model.eval()
    ys, ps = [], []
    for xb, yb, m in loader:
        xb, yb, m = xb.to(device), yb.to(device), m.to(device)
        pred = model(xb, key_padding_mask=m).argmax(1)
        ys.append(yb.cpu().numpy()); ps.append(pred.cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    out = {}
    if "acc" in metrics: out["acc"] = float(accuracy_score(y_true, y_pred))
    if "macro_f1" in metrics: out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    if "kappa" in metrics: out["kappa"] = float(cohen_kappa_score(y_true, y_pred))
    return out

def early_stop_update(best_metric, current_metric, patience_cnt, mode="max"):
    improve = (current_metric > best_metric) if mode=="max" else (current_metric < best_metric)
    if improve: return current_metric, 0, True
    return best_metric, patience_cnt+1, False

def save_csv_summary(path: Path, ks: List[int], metrics: List[str], agg: Dict[int, Dict[str,float]]):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["k"] + [f"{m}_mean" for m in metrics] + [f"{m}_std" for m in metrics]
        w.writerow(header)
        for k in ks:
            row = [k] + [agg[k][f"{m}_mean"] for m in metrics] + [agg[k][f"{m}_std"] for m in metrics]
            w.writerow(row)

from collections import defaultdict

def get_label_for_pid(pid: str,
                      labels_map: Optional[Dict[str,int]],
                      code2idx: Optional[Dict[str,int]]) -> Optional[int]:
    if labels_map and str(pid) in labels_map:
        return labels_map[str(pid)]
    return code2idx_label_for_pid(str(pid), code2idx)

def class_counts_from_pids(pids: List[str],
                           labels_map: Optional[Dict[str,int]],
                           code2idx: Optional[Dict[str,int]]) -> Dict[int,int]:
    cnt = defaultdict(int)
    for pid in pids:
        y = get_label_for_pid(str(pid), labels_map, code2idx)
        if y is not None:
            cnt[y] += 1
    return dict(cnt)

def filter_pids_by_classes(pids: List[str],
                           labels_map: Optional[Dict[str,int]],
                           code2idx: Optional[Dict[str,int]],
                           keep_classes: set) -> List[str]:
    out = []
    for pid in pids:
        y = get_label_for_pid(str(pid), labels_map, code2idx)
        if (y is not None) and (y in keep_classes):
            out.append(pid)
    return out

    
def enforce_k_shot(pids: List[str],
                   labels_map: Optional[Dict[str,int]],
                   code2idx: Optional[Dict[str,int]],
                   k: int,
                   seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    buckets: Dict[int, List[str]] = {}
    for pid in pids:
        if labels_map and str(pid) in labels_map:
            y = labels_map[str(pid)]
        else:
            y = code2idx_label_for_pid(str(pid), code2idx)
        if y is None:
            raise AssertionError(f"Missing label for pid={pid} when enforcing k-shot.")
        buckets.setdefault(y, []).append(str(pid))

    selected = []
    for y, plist in buckets.items():
        if len(plist) < k:
            raise AssertionError(f"class {y} has only {len(plist)} samples, need k={k}.")
        plist_sorted = sorted(plist)
        rng.shuffle(plist_sorted)
        selected.extend(plist_sorted[:k])

    selected_set = set(selected)
    return [pid for pid in pids if str(pid) in selected_set]

def build_code_index_no_test(split_paths: Iterable[str]) -> Dict[str,int]:
    codes = set()
    for sp in split_paths:
        if not sp: continue
        p = Path(sp)
        if not p.is_file():
            continue
        splits, labels = load_split_and_labels(sp)
        for subset in ("train","val"):
            for pid in splits.get(subset, []):
                if labels and str(pid) in labels:
                    continue
                c = extract_label_code_from_pid(str(pid))
                if c is not None:
                    codes.add(c)
    unique = sorted(codes)
    return {c:i for i,c in enumerate(unique)}


# ----------------- main -----------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(cfg.get("device", "cuda"))
    seeds = cfg["eval"].get("seeds", [42])
    debug_prints = bool(cfg.get("debug_prints", True))

    ds_cfg = cfg["dataset"]
    root_npz = ds_cfg["root_npz"]
    drop_b10 = bool(ds_cfg.get("drop_b10", True))
    normalize = ds_cfg.get("normalize", "zscore")

    enc = cfg["encoder"]
    d_model = int(enc.get("dim", 256))
    layers  = int(enc.get("layers", 2))
    nhead   = int(enc.get("nhead", 8))
    ff_mult = int(enc.get("ff_mult", 4))
    dropout = float(enc.get("dropout", 0.1))
    max_len = int(enc.get("max_len", 1024))

    pt = cfg["pretraining"]
    scenario = pt.get("scenario", "LV_overlap")
    pretrain_map = {
        "LV": ds_cfg["pretrain_json"],
        "LV_overlap": ds_cfg.get("pretrain_json_overlap", ds_cfg["pretrain_json"]),
        "LVPT": ds_cfg.get("pretrain_json_lvpt", ds_cfg["pretrain_json"]),
        "LVPT_overlap": ds_cfg.get("pretrain_json_lvpt_overlap", ds_cfg["pretrain_json"]),
    }
    base_pretrain_json = pretrain_map[scenario]
    pt_bs = int(pt.get("batch_size", 128))
    pt_epochs = int(pt.get("epochs", 150))
    pt_pat = int(pt.get("early_stopping_patience", 15))
    pt_lr = float(pt.get("lr", 3e-4))
    pt_wd = float(pt.get("weight_decay", 1e-4))
    pt_cos = pt.get("cosine_anneal", {"enabled": True, "cycles": 1})
    do_downsample = bool(pt.get("downsample_meadow_to_median", True))
    meadow_id = pt.get("meadow_label_id", None)

    ft = cfg["finetune"]
    k_grid = list(ft.get("k_grid", [1,5,10,20,100,200,500]))
    ft_bs = int(ft.get("batch_size", 16))
    ft_epochs = int(ft.get("epochs", 200))
    ft_pat = int(ft.get("early_stopping_patience", 9999))
    lr_mode_yaml = ft.get("lr_mode", "head_only")
    ft_wd = float(ft.get("weight_decay", 1e-4))
    ft_lr_all = float(ft.get("lr", {}).get("same", 3e-4))
    ft_lr_head_only = float(ft.get("lr", {}).get("head_only", 3e-4))
    ft_lr_sep_head = float(ft.get("lr", {}).get("separate", {}).get("head", 3e-4))
    ft_lr_sep_back = float(ft.get("lr", {}).get("separate", {}).get("backbone", 1e-4))
    ft_cos = ft.get("cosine_anneal", {"enabled": False})

    metrics = cfg["eval"].get("metrics", ["acc"])
    outdir = Path(cfg["paths"]["outputs_dir"]); outdir.mkdir(parents=True, exist_ok=True)

    # ---------- helpers for strict k-shot ----------
    from collections import defaultdict
    def get_label_for_pid(pid: str,
                          labels_map: Optional[Dict[str,int]],
                          code2idx: Optional[Dict[str,int]]) -> Optional[int]:
        if labels_map and str(pid) in labels_map:
            return labels_map[str(pid)]
        return code2idx_label_for_pid(str(pid), code2idx)

    def class_counts_from_pids(pids: List[str],
                               labels_map: Optional[Dict[str,int]],
                               code2idx: Optional[Dict[str,int]]) -> Dict[int,int]:
        cnt = defaultdict(int)
        for pid in pids:
            y = get_label_for_pid(str(pid), labels_map, code2idx)
            if y is not None:
                cnt[y] += 1
        return dict(cnt)

    def filter_pids_by_classes(pids: List[str],
                               labels_map: Optional[Dict[str,int]],
                               code2idx: Optional[Dict[str,int]],
                               keep_classes: set) -> List[str]:
        out = []
        for pid in pids:
            y = get_label_for_pid(str(pid), labels_map, code2idx)
            if (y is not None) and (y in keep_classes):
                out.append(pid)
        return out
    # ----------------------------------------------

    # Build code->idx using only pretrain & FT train/val (never test)
    split_paths = [base_pretrain_json] + [ds_cfg["finetune_json_tpl"].format(k=k) for k in k_grid]
    code2idx = build_code_index_no_test(split_paths)

    # infer input channels from *base* pretrain JSON
    in_dim = infer_in_dim(root_npz, base_pretrain_json, drop_b10)

    all_results = []
    for seed in seeds:
        set_seed(seed)
        sdir = outdir / f"seed_{seed}"; sdir.mkdir(parents=True, exist_ok=True)
        results_seed = {"seed": seed, "per_k": {}}

        for k in k_grid:
            # ----- resolve FT split for this k
            ft_json = ds_cfg["finetune_json_tpl"].format(k=k)
            sp_ft, labels_ft = load_split_and_labels(ft_json)

            # integrity checks
            assert_disjoint(sp_ft["train"], sp_ft["val"], f"FT-train(k={k})", f"FT-val(k={k})")
            assert_disjoint(sp_ft["train"], sp_ft["test"], f"FT-train(k={k})", f"FT-test(k={k})")
            assert_disjoint(sp_ft["val"],   sp_ft["test"], f"FT-val(k={k})",   f"FT-test(k={k})")

            # ---------- strict k-shot: keep only classes with >= k in FT-train, filter all splits ----------
            train_counts = class_counts_from_pids(sp_ft["train"], labels_ft, code2idx)
            eligible = {y for y, c in train_counts.items() if c >= k}
            if len(eligible) == 0:
                raise AssertionError(f"No classes have >= {k} samples in FT train; cannot run k-shot={k}.")

            orig_sizes = {s: len(sp_ft[s]) for s in ("train","val","test")}
            sp_ft["train"] = filter_pids_by_classes(sp_ft["train"], labels_ft, code2idx, eligible)
            sp_ft["val"]   = filter_pids_by_classes(sp_ft["val"],   labels_ft, code2idx, eligible)
            sp_ft["test"]  = filter_pids_by_classes(sp_ft["test"],  labels_ft, code2idx, eligible)

            if debug_prints:
                print(f"[k={k}] strict-k filter: eligible classes={len(eligible)} | "
                      f"train {orig_sizes['train']}→{len(sp_ft['train'])}, "
                      f"val {orig_sizes['val']}→{len(sp_ft['val'])}, "
                      f"test {orig_sizes['test']}→{len(sp_ft['test'])}")

            # now enforce exactly k per class on the filtered train list
            train_pids_k = enforce_k_shot(list(sp_ft["train"]), labels_ft, code2idx, k, seed=seed)
            # -----------------------------------------------------------------------------------------------

            # ----- build LV* for this k (EE train+val classes only), then PRETRAIN
            splits_pt_base, labels_pt = load_split_and_labels(base_pretrain_json)
            pt_train_all = list(splits_pt_base["train"])
            pt_val_all   = list(splits_pt_base["val"])

            ee_tv_codes = _codes_from_pids(sp_ft["train"]) | _codes_from_pids(sp_ft["val"])
            before = len(pt_train_all)
            pt_train = [pid for pid in pt_train_all if extract_label_code_from_pid(str(pid)) in ee_tv_codes]
            if do_downsample and (labels_pt is not None) and (meadow_id is not None):
                pt_train = downsample_class_to_median(pt_train, labels_pt, int(meadow_id))
            if debug_prints:
                print(f"[seed {seed}] k={k} LV* filter (EE train+val classes): {before} -> {len(pt_train)}")

            # no PID leakage pretrain vs FT
            assert_disjoint(pt_train + pt_val_all, sp_ft["train"] + sp_ft["val"] + sp_ft["test"],
                            f"PRETRAIN(k={k})", f"FT(k={k})")

            # pretrain loaders
            ds_pt_tr = SeqDataset(root_npz, base_pretrain_json, "train",
                                  drop_b10=drop_b10, normalize=normalize,
                                  only_pids=pt_train, labels_by_pid=labels_pt, code2idx=code2idx)
            ds_pt_va = SeqDataset(root_npz, base_pretrain_json, "val",
                                  drop_b10=drop_b10, normalize=normalize,
                                  labels_by_pid=labels_pt, code2idx=code2idx)
            ld_pt_tr = DataLoader(ds_pt_tr, batch_size=pt_bs, shuffle=True,  num_workers=cfg.get("num_workers",8),
                                  pin_memory=True, drop_last=False, collate_fn=collate_seq)
            ld_pt_va = DataLoader(ds_pt_va, batch_size=pt_bs, shuffle=False, num_workers=cfg.get("num_workers",8),
                                  pin_memory=True, drop_last=False, collate_fn=collate_seq)

            # model + pretrain
            n_classes_pre = infer_num_classes_from_pids(ds_pt_va.pids, labels_pt, root_npz, code2idx)
            model = TinyTSFormer(in_channels=in_dim, n_classes=n_classes_pre,
                                 d_model=d_model, nhead=nhead, nlayers=layers,
                                 ff_mult=ff_mult, dropout=dropout, max_len=max_len).to(device)

            criterion = nn.CrossEntropyLoss()
            opt = torch.optim.AdamW(model.parameters(), lr=pt_lr, weight_decay=pt_wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, pt_epochs // int(pt_cos.get("cycles", 1)))
            ) if pt_cos.get("enabled", True) else None

            best_val, best_state, patience = -1.0, None, 0
            for ep in range(1, pt_epochs+1):
                model.train()
                for xb, yb, m in ld_pt_tr:
                    xb, yb, m = xb.to(device), yb.to(device), m.to(device)
                    opt.zero_grad(); loss = criterion(model(xb, key_padding_mask=m), yb)
                    loss.backward(); opt.step()
                if sched is not None: sched.step()
                val_acc = evaluate(model, ld_pt_va, device, metrics=("acc",))["acc"]
                best_val, patience, improved = early_stop_update(best_val, val_acc, patience, "max")
                if improved: best_state = {kk: v.detach().cpu() for kk, v in model.state_dict().items()}
                if patience >= pt_pat: break
            if best_state is not None:
                model.load_state_dict({kk: v.to(device) for kk, v in best_state.items()})
            torch.save(model.state_dict(), sdir / f"pretrained_k{k}.pt")

            # ----- unseen-test filter (after strict-k) ----------
            train_codes = _codes_from_pids(sp_ft["train"])
            val_codes   = _codes_from_pids(sp_ft["val"])
            test_codes  = _codes_from_pids(sp_ft["test"])
            trainval_codes = train_codes | val_codes
            unseen_test = test_codes - trainval_codes
            if unseen_test:
                before_t = len(sp_ft["test"])
                sp_ft["test"] = [pid for pid in sp_ft["test"]
                                 if extract_label_code_from_pid(str(pid)) in trainval_codes]
                print(f"[warn] filtered {before_t - len(sp_ft['test'])} test samples with unseen classes "
                      f"(e.g., {list(sorted(unseen_test))[:5]})")

            # ----- FT datasets/loaders
            ds_tr = SeqDataset(root_npz, ft_json, "train",
                               drop_b10=drop_b10, normalize=normalize,
                               only_pids=train_pids_k,                 # enforced k-shot
                               labels_by_pid=labels_ft, code2idx=code2idx)
            ds_va = SeqDataset(root_npz, ft_json, "val",
                               drop_b10=drop_b10, normalize=normalize,
                               labels_by_pid=labels_ft, code2idx=code2idx)
            ds_te = SeqDataset(root_npz, ft_json, "test",
                               drop_b10=drop_b10, normalize=normalize,
                               only_pids=list(sp_ft["test"]),          # filtered test PIDs
                               labels_by_pid=labels_ft, code2idx=code2idx)

            ld_tr = DataLoader(ds_tr, batch_size=ft_bs, shuffle=True,  num_workers=cfg.get("num_workers",8),
                               pin_memory=True, drop_last=False, collate_fn=collate_seq)
            ld_va = DataLoader(ds_va, batch_size=ft_bs, shuffle=False, num_workers=cfg.get("num_workers",8),
                               pin_memory=True, drop_last=False, collate_fn=collate_seq)
            ld_te = DataLoader(ds_te, batch_size=ft_bs, shuffle=False, num_workers=cfg.get("num_workers",8),
                               pin_memory=True, drop_last=False, collate_fn=collate_seq)

            # reset head to EE classes, then FT
            n_classes_ee = infer_num_classes_from_pids(ds_te.pids, labels_ft, root_npz, code2idx)
            model.head = nn.Linear(d_model, n_classes_ee).to(device)

            lr_mode_eff = "head_only" if k <= 20 else lr_mode_yaml
            if lr_mode_eff == "head_only":
                for p in model.parameters(): p.requires_grad = False
                for p in model.head.parameters(): p.requires_grad = True
                opt = torch.optim.AdamW([{"params": model.head.parameters(), "lr": ft_lr_head_only}], weight_decay=ft_wd)
            elif lr_mode_eff == "same":
                for p in model.parameters(): p.requires_grad = True
                opt = torch.optim.AdamW([{"params": model.parameters(), "lr": ft_lr_all}], weight_decay=ft_wd)
            else:
                for p in model.parameters(): p.requires_grad = True
                opt = torch.optim.AdamW([
                    {"params": (p for n,p in model.named_parameters() if not n.startswith("head.")), "lr": ft_lr_sep_back},
                    {"params": model.head.parameters(), "lr": ft_lr_sep_head},
                ], weight_decay=ft_wd)

            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, ft_epochs)
            ) if ft_cos.get("enabled", False) else None
            criterion = nn.CrossEntropyLoss()

            for ep in range(1, ft_epochs+1):
                model.train()
                for xb, yb, m in ld_tr:
                    xb, yb, m = xb.to(device), yb.to(device), m.to(device)
                    opt.zero_grad(); loss = criterion(model(xb, key_padding_mask=m), yb)
                    loss.backward(); opt.step()
                if sched is not None: sched.step()
                _ = evaluate(model, ld_va, device, metrics=("acc",))  # monitor only

            test_metrics = evaluate(model, ld_te, device, metrics=metrics)
            results_seed["per_k"][k] = test_metrics
            with open(sdir / f"results_k{k}.json", "w") as f:
                json.dump(test_metrics, f, indent=2)
            if debug_prints:
                print(f"[seed {seed}] k={k} test:", test_metrics)

        all_results.append(results_seed)

    # aggregate & save
    ks = sorted(k_grid)
    agg = {}
    for k in ks:
        bucket = {m: [] for m in metrics}
        for r in all_results:
            tm = r["per_k"][k]
            for m in metrics: bucket[m].append(tm[m])
        agg[k] = {}
        for m, vals in bucket.items():
            agg[k][f"{m}_mean"] = float(np.mean(vals))
            agg[k][f"{m}_std"]  = float(np.std(vals))

    with open(outdir / "summary.json", "w") as f:
        json.dump({"seeds": seeds, "k_grid": ks, "aggregate": agg, "per_seed": all_results}, f, indent=2)
    save_csv_summary(outdir / "summary.csv", ks, metrics, agg)

    print("\n=== Aggregate (mean ± std over seeds) ===")
    for k in ks:
        row = [f"k={k}"] + [f"{m}={agg[k][f'{m}_mean']:.3f}±{agg[k][f'{m}_std']:.3f}" for m in metrics]
        print("  " + "  ".join(row))


if __name__ == "__main__":
    main()
