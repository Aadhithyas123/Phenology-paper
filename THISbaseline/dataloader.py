# src/dataloader.py
import os
import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Iterable, Union, Any
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logger = logging.getLogger("eurocropsml.dataloader")
if not logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# =========================================================
#                NPZ LOADING & LABEL PARSING
# =========================================================

def _load_npz_timeseries(path: str,
                         expect_13_or_12_channels: bool = True,
                         drop_b10: bool = False) -> np.ndarray:
    """
    Load a EuroCrops-style .npz and return float32 array [T, C].

    Accepts time-series keys: 'ts', 'timeseries', 'x', 'X', 'data'
    Accepts optional 'dates' key (array-like, length T).
    Will coerce to [T, C] and warn/log on suspicious shapes.

    Args
    ----
    path : str
    expect_13_or_12_channels : if True, warns if C not in {12, 13}
    drop_b10 : if True, and C == 13, drops band B10 (index 9 in B1..B12 order)

    Returns
    -------
    ts : np.ndarray [T, C] (float32)
    """
    with np.load(path, allow_pickle=True) as z:
        # --- locate time series
        ts = None
        for k in ("ts", "timeseries", "x", "X", "data"):
            if k in z:
                ts = np.asarray(z[k])
                break
        if ts is None:
            raise KeyError(f"No expected time-series key in {path}. Found: {list(z.keys())}")

        # --- squeeze trivial dims like [T,C,1] or [1,T,C]
        if ts.ndim == 3 and 1 in ts.shape:
            ts = np.squeeze(ts)

        # --- coerce to [T, C] using 'dates' if available
        dates = None
        if "dates" in z:
            dates = np.array(z["dates"])
        if ts.ndim != 2:
            if dates is not None:
                nd = int(len(dates))
                if ts.shape[0] == nd:
                    ts = ts.reshape(nd, -1)
                elif ts.shape[-1] == nd:
                    ts = ts.reshape(-1, nd).T
                else:
                    raise ValueError(f"Can't reshape to [T,C]; got {ts.shape} with {nd} dates in {path}")
            else:
                raise ValueError(f"Expected 2D array, got {ts.shape} in {path}")

        T, C = ts.shape

        # Heuristic for S2: if channels=13 is on axis 0, transpose
        if C != 13 and T == 13:
            ts = ts.T
            T, C = ts.shape

        # Optional: drop B10
        if drop_b10 and C == 13:
            # B10 at index 9 (0-based) in B1..B12 order
            ts = np.delete(ts, 9, axis=1)
            C -= 1

        if expect_13_or_12_channels and C not in (12, 13):
            logger.warning(f"{os.path.basename(path)}: unexpected channel count C={C} (expected 12 or 13)")

        if T <= 0:
            raise ValueError(f"{os.path.basename(path)}: empty time axis (T=0)")

        return ts.astype(np.float32)


def _load_npz_full(path: str,
                   expect_13_or_12_channels: bool = True,
                   drop_b10: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load [T,C] and dates (if present). Performs same normalization as _load_npz_timeseries.
    """
    with np.load(path, allow_pickle=True) as z:
        dates = None
        if "dates" in z:
            dates = np.array(z["dates"])
        # reuse path through _load_npz_timeseries with same logic
    ts = _load_npz_timeseries(path, expect_13_or_12_channels=expect_13_or_12_channels, drop_b10=drop_b10)
    with np.load(path, allow_pickle=True) as z:
        if "dates" in z:
            dates = np.array(z["dates"])
    return ts, dates


HCAT3_RE = re.compile(r".*_(\d{10})\.npz$")  # final 10-digit code

def parse_hcat3_from_fname(fname: str) -> str:
    base = os.path.basename(fname)
    m = HCAT3_RE.match(base)
    if m:
        return m.group(1)
    toks = os.path.splitext(base)[0].split("_")
    for tok in toks:
        if tok.isdigit() and len(tok) == 10:
            return tok
    raise ValueError(f"Cannot parse HCAT3 from filename: {fname}")

def parse_country_from_fname(fname: str) -> str:
    toks = os.path.splitext(os.path.basename(fname))[0].split("_")
    if not toks:
        raise ValueError(f"Bad filename: {fname}")
    return toks[0][:2].upper()

def parse_label_from_fname(fname: str) -> str:
    # alias: crop code = HCAT3 for now
    return parse_hcat3_from_fname(fname)

def normalize_label(code: str) -> str:
    code = str(code).strip()
    return code  # hook if you later collapse label levels


# =========================================================
#                     SPLIT READING
# =========================================================

def _fix_path(root_npz: str, rel_or_abs: str) -> str:
    return rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(root_npz, rel_or_abs)

def _rows_from_json_block(block: Any, default_label_from_fname: bool = True) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    if isinstance(block, list):
        for item in block:
            if isinstance(item, str):
                lab = parse_label_from_fname(item) if default_label_from_fname else None
                rows.append({"fname": item, "label": lab})
            elif isinstance(item, dict):
                fname = item.get("fname") or item.get("path") or item.get("file")
                if fname is None:
                    for v in item.values():
                        if isinstance(v, str) and (v.endswith(".npz") or "/" in v or "\\" in v):
                            fname = v
                            break
                label = item.get("label")
                if label is None and default_label_from_fname and fname:
                    label = parse_label_from_fname(fname)
                if fname is None:
                    raise ValueError(f"JSON item missing 'fname': {item}")
                rows.append({"fname": fname, "label": label})
            else:
                raise ValueError(f"Unsupported JSON list element: {type(item)}")

    elif isinstance(block, dict):
        for k, v in block.items():
            if isinstance(v, list):
                use_k_as_label = bool(re.fullmatch(r"\d{4,}", str(k)))
                for item in v:
                    if isinstance(item, str):
                        rows.append({
                            "fname": item,
                            "label": str(k) if use_k_as_label else (
                                parse_label_from_fname(item) if default_label_from_fname else None
                            )
                        })
                    elif isinstance(item, dict):
                        fname = item.get("fname") or item.get("path") or item.get("file")
                        label = item.get("label")
                        if label is None and use_k_as_label:
                            label = str(k)
                        if label is None and default_label_from_fname and fname:
                            label = parse_label_from_fname(fname)
                        if fname is None:
                            raise ValueError(f"JSON dict-block item missing 'fname': {item}")
                        rows.append({"fname": fname, "label": label})
                    else:
                        raise ValueError(f"Unsupported JSON dict-list element: {type(item)}")
            else:
                rows.extend(_rows_from_json_block(v, default_label_from_fname=default_label_from_fname))
    else:
        raise ValueError(f"Unsupported JSON block type: {type(block)}")

    return rows


def read_split_to_df(split_file: str,
                     split_subset: Optional[Union[str, Iterable[str]]] = None,
                     countries: Optional[Iterable[str]] = None,
                     only_overlap_with: Optional[Iterable[str]] = None,
                     unknown_label_list: Optional[Iterable[str]] = None) -> pd.DataFrame:

    if split_file.endswith(".parquet"):
        df = pd.read_parquet(split_file)

    elif split_file.endswith(".json"):
        with open(split_file, "r") as f:
            obj = json.load(f)

        if split_subset is None:
            split_subset = ["train"]
        elif isinstance(split_subset, str):
            split_subset = [split_subset]

        rows: List[Dict[str, str]] = []
        if isinstance(obj, dict):
            for part in split_subset:
                if part in obj:
                    rows.extend(_rows_from_json_block(obj[part], default_label_from_fname=True))
        elif isinstance(obj, list):
            rows.extend(_rows_from_json_block(obj, default_label_from_fname=True))
        else:
            raise ValueError(f"Unsupported JSON root type: {type(obj)}")

        if not rows:
            raise ValueError(f"No files found in {split_file} for subsets {split_subset}")

        df = pd.DataFrame(rows)

    else:
        df = pd.read_csv(split_file)

    if "fname" not in df.columns:
        raise ValueError("Split file must contain 'fname'.")

    # fill labels if missing
    if "label" not in df.columns or df["label"].isnull().any() or (df["label"] == "").any():
        df["label"] = df["fname"].map(parse_label_from_fname)

    df["hcat3"] = df["fname"].map(parse_hcat3_from_fname)
    df["country"] = df["fname"].map(parse_country_from_fname)
    df["label"] = df["label"].astype(str).map(normalize_label)

    # country filter
    if countries:
        cset = {c.upper() for c in countries}
        df = df[df["country"].str.upper().isin(cset)]

    # overlap filter
    if only_overlap_with:
        need_countries = {c.upper() for c in only_overlap_with}
        per_cty = df.groupby("country")["label"].apply(lambda s: set(s.astype(str))).to_dict()
        if need_countries.issubset(set(per_cty.keys())):
            overlap = set.intersection(*[per_cty[c] for c in need_countries])
            df = df[df["label"].astype(str).isin(overlap)]

    df["is_unknown"] = False
    if unknown_label_list:
        unk = {str(u) for u in unknown_label_list}
        df.loc[df["label"].astype(str).isin(unk), "is_unknown"] = True

    return df.reset_index(drop=True)


# =========================================================
#                         DATASET
# =========================================================

import os
import logging
from typing import Dict, List, Optional, Iterable, Union, Any
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Expect these helpers to already exist in your file:
# from .dataloader import _load_npz_timeseries, read_split_to_df, _fix_path

log = logging.getLogger("eurocropsml.dataloader")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)


class EuroCropsFSL(Dataset):
    """
    EuroCrops Few-Shot Learning dataset.

    Builds a dataframe of parcels from a split file, applies optional filters
    (countries, overlap label whitelist), constructs/uses label2idx, and
    yields padded variable-length time series for few-shot episodic sampling.

    Key safety features:
      - label_whitelist: enforce overlap classes across domains *before* mapping
      - safe label2idx mapping: unmapped labels are dropped before astype(int)
      - clear errors if dataset becomes empty

    Args
    ----
    root_npz : str
        Root directory containing per-parcel .npz time series.
    split_file : str
        JSON/CSV/Parquet describing the split(s).
    countries : Optional[Iterable[str]]
        Keep only rows whose parsed country belongs to this set.
    only_overlap_with : Optional[Iterable[str]]
        If provided, keep only labels that appear in *all* these countries
        (computed inside read_split_to_df). Often you will instead compute a
        whitelist externally and pass via label_whitelist.
    label_map : Optional[Dict[str,str]]
        Optional remapping of labels (e.g., parent↔leaf).
    unknown_label_list : Optional[Iterable[str]]
        Marks rows as unknown; useful for open-set logic elsewhere.
    leaf2parent : Optional[Dict[str,str]]
        Mapping from leaf_id to parent (for hierarchy reporting).
    split_subset : Optional[Union[str, Iterable[str]]]
        "train"/"val"/"test"/None. If None, read all and use country filter.
    label2idx : Optional[Dict[str,int]]
        If given (e.g., from training set), val/test labels must map into it.
    auto_extend_labels : bool
        If True and label2idx is provided, unseen labels are appended to map.
    inplace_extend : bool
        If False and label2idx is provided, a copy of the map is extended.
    normalize : str
        "zscore" | "minmax" | "none".
    norm_stats : Optional[Dict[str,np.ndarray]]
        Pre-computed normalization stats. If None and normalize != "none",
        stats are estimated on a stratified subsample.
    time_clip : Optional[int]
        If set, truncate time dimension to this many steps from the start.
    kshot_cap_per_class : Optional[int]
        If set, downsample each class to at most this many items.
    seed : int
        RNG seed for any sampling/normalization subsampling.
    label_whitelist : Optional[Iterable[str]]
        Set of allowed labels (strings). Enforced *before* label2idx mapping.
    drop_b10 : bool
        If True and C == 13, drop B10 (index 10 assuming canonical order).
    """

    def __init__(self,
                 root_npz: str,
                 split_file: str,
                 countries: Optional[Iterable[str]] = None,
                 only_overlap_with: Optional[Iterable[str]] = None,
                 label_map: Optional[Dict[str, str]] = None,
                 unknown_label_list: Optional[Iterable[str]] = None,
                 leaf2parent: Optional[Dict[str, str]] = None,
                 split_subset: Optional[Union[str, Iterable[str]]] = None,
                 label2idx: Optional[Dict[str, int]] = None,
                 auto_extend_labels: bool = True,
                 inplace_extend: bool = True,
                 normalize: str = "zscore",
                 norm_stats: Optional[Dict[str, np.ndarray]] = None,
                 time_clip: Optional[int] = None,
                 kshot_cap_per_class: Optional[int] = None,
                 seed: int = 42,
                 label_whitelist: Optional[Iterable[str]] = None,
                 drop_b10: bool = False):

        super().__init__()
        self.root_npz = root_npz
        self.normalize = normalize
        self.norm_stats = norm_stats
        self.time_clip = time_clip
        self.drop_b10 = bool(drop_b10)
        self._rng = np.random.RandomState(int(seed))

        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cache_cap = 4096

        # 1) Read split → DataFrame with at least [fname, label, country, hcat3]
        df = read_split_to_df(
            split_file=split_file,
            split_subset=split_subset,
            countries=countries,
            only_overlap_with=only_overlap_with,
            unknown_label_list=unknown_label_list
        )

        # 2) Absolute path column
        df["path"] = df["fname"].map(lambda s: _fix_path(root_npz, s))

        # 3) Optional label remap
        if label_map:
            df["label"] = df["label"].astype(str).map(lambda c: label_map.get(str(c), str(c)))

        # 4) Keep both leaf and parent; rely on provided mapping if any
        df["leaf_id"] = df["label"].astype(str)
        if leaf2parent is not None:
            df["parent_id"] = df["leaf_id"].map(leaf2parent).astype(str)
        else:
            # fallback: use parsed hcat3 from filename
            df["parent_id"] = df["hcat3"].astype(str)

        # 5) Enforce label whitelist (overlap) BEFORE building label2idx
        if label_whitelist is not None:
            w = {str(x) for x in label_whitelist}
            before = len(df)
            df = df[df["leaf_id"].astype(str).isin(w)].copy()
            dropped = before - len(df)
            if dropped > 0:
                log.info(f"[EuroCropsFSL] label_whitelist dropped {dropped} rows (kept={len(df)})")

        # 6) Optional per-class cap (k-shot pretraining downsample, etc.)
        if kshot_cap_per_class is not None:
            capped = []
            for _, g in df.groupby("leaf_id", sort=False):
                if len(g) > kshot_cap_per_class:
                    g = g.sample(n=kshot_cap_per_class, random_state=int(seed))
                capped.append(g)
            df = pd.concat(capped, axis=0).reset_index(drop=True)

        # 7) Build/use label2idx safely
        labels = df["leaf_id"].astype(str).tolist()
        if label2idx is None:
            uniq = sorted(set(labels))
            self.label2idx = {lab: i for i, lab in enumerate(uniq)}
        else:
            self.label2idx = label2idx if inplace_extend else dict(label2idx)
            if auto_extend_labels:
                missing = sorted({lab for lab in labels if lab not in self.label2idx})
                if missing:
                    start = len(self.label2idx)
                    for i, lab in enumerate(missing):
                        self.label2idx[lab] = start + i
            else:
                # STRICT: drop rows not in provided training label2idx
                m = df["leaf_id"].astype(str).isin(self.label2idx.keys())
                before = len(df)
                df = df[m].copy()
                dropped = before - len(df)
                if dropped > 0:
                    log.info(f"[EuroCropsFSL] dropped {dropped} rows not in training label2idx.")

        # 8) Safe mapping → y (drop any residual NaNs)
        df["y"] = df["leaf_id"].map(self.label2idx)
        n_nan = int(df["y"].isna().sum())
        if n_nan > 0:
            log.warning(f"[EuroCropsFSL] {n_nan} unmapped rows after map(); dropping.")
            df = df[~df["y"].isna()].copy()

        if len(df) == 0:
            raise RuntimeError("EuroCropsFSL ended up empty after filtering/mapping. "
                               "Check split file, countries, and overlap whitelist.")

        df["y"] = df["y"].astype(int)

        # 9) Final slim DF held by the dataset
        self.df = df[["path", "y", "leaf_id", "parent_id", "hcat3", "country", "is_unknown"]].reset_index(drop=True)

        # 10) Build class_to_indices and frequencies
        self.class_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, y in enumerate(self.df["y"].tolist()):
            self.class_to_indices[int(y)].append(idx)
        self.class_to_indices = dict(self.class_to_indices)
        self.num_classes = len(self.class_to_indices)
        self.freq = self.df["y"].value_counts().sort_index()

        # 11) Fit normalization stats if needed
        if self.normalize != "none" and self.norm_stats is None:
            self.norm_stats = self._fit_normalizer(sample_per_class=64)

        log.info(f"Dataset ready: N={len(self.df)}, classes={len(self.class_to_indices)}, "
                 f"normalize={self.normalize}, drop_b10={self.drop_b10}")

    # ---------------- normalization helpers ----------------

    def _fit_normalizer(self, sample_per_class: int = 64) -> Dict[str, np.ndarray]:
        samples = []
        for _, idxs in self.class_to_indices.items():
            take = idxs if len(idxs) <= sample_per_class else self._rng.choice(idxs, sample_per_class, replace=False)
            for i in take:
                x = self._load_series_by_row_index(i)  # [T, C]
                samples.append(x)
        if not samples:
            return {}

        all_ts = np.concatenate(samples, axis=0)  # [sumT, C]
        if self.normalize == "zscore":
            mean = np.nanmean(all_ts, axis=0)
            std = np.nanstd(all_ts, axis=0) + 1e-8
            return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
        elif self.normalize == "minmax":
            vmin = np.nanmin(all_ts, axis=0)
            vmax = np.nanmax(all_ts, axis=0)
            rng = np.maximum(vmax - vmin, 1e-8)
            return {"min": vmin.astype(np.float32), "rng": rng.astype(np.float32)}
        return {}

    def _apply_norm(self, x: np.ndarray) -> np.ndarray:
        if self.normalize == "none" or not self.norm_stats:
            return x
        if self.normalize == "zscore":
            return (x - self.norm_stats["mean"]) / self.norm_stats["std"]
        elif self.normalize == "minmax":
            return (x - self.norm_stats["min"]) / self.norm_stats["rng"]
        return x

    # ---------------- internals ----------------

    def _load_series_by_row_index(self, i: int) -> np.ndarray:
        """
        Load, time-clip, optionally drop B10, and return float32 [T, C].
        Caching keeps recently used arrays.
        """
        if i in self._cache:
            x = self._cache[i]
        else:
            path = self.df.iloc[i]["path"]
            x = _load_npz_timeseries(path)  # [T, C]
            if self._cache_cap > 0:
                self._cache[i] = x
                if len(self._cache) > self._cache_cap:
                    # pop oldest
                    self._cache.pop(next(iter(self._cache)))
        # time clip
        if self.time_clip is not None:
            x = x[: int(self.time_clip)]
        # drop B10 if requested and 13-band input
        if self.drop_b10 and x.ndim == 2 and x.shape[1] == 13:
            # canonical S2 order assumed: B01..B12 with B10 at index 10 (0-based)
            if x.shape[1] > 10:
                x = np.concatenate([x[:, :10], x[:, 11:]], axis=1)  # remove column 10
        return x

    # ---------------- torch dataset API ----------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self._load_series_by_row_index(idx)       # [T, C]
        x = self._apply_norm(x)
        T, _ = x.shape
        return {
            "x": torch.from_numpy(x.copy()),          # [T, C] float32
            "y": torch.tensor(int(self.df.iloc[idx]["y"])),  # scalar long
            "len": torch.tensor(T),                   # scalar long
            "path": self.df.iloc[idx]["path"],        # str
        }


# =========================================================
#                     COLLATE WITH PADDING
# =========================================================

def pad_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    lengths = torch.tensor([b["x"].shape[0] for b in batch], dtype=torch.long)
    C = batch[0]["x"].shape[1]
    Tmax = int(lengths.max().item())
    B = len(batch)

    x_pad = torch.zeros((B, Tmax, C), dtype=torch.float32)
    mask = torch.zeros((B, Tmax), dtype=torch.bool)
    y = torch.tensor([int(b["y"]) for b in batch], dtype=torch.long)

    for i, b in enumerate(batch):
        T = b["x"].shape[0]
        x_pad[i, :T] = b["x"]
        mask[i, :T] = True

    return {"x": x_pad, "mask": mask, "y": y, "lengths": lengths}


# =========================================================
#                FEW-SHOT EPISODE SAMPLER
# =========================================================

class FewShotEpisodeSampler(Sampler[List[int]]):
    def __init__(self,
                 dataset: EuroCropsFSL,                 # EuroCropsFSL or DatasetView
                 n_way: int,
                 k_shot: int,
                 q_query: int,
                 episodes_per_epoch: int,
                 rng_seed: int = 0,
                 sampling_mode: str = "balanced",   # "balanced" | "real" | "tail_mix"
                 inject_unknown_ratio: float = 0.0  # fraction of queries replaced by unknowns
                 ):
        super().__init__(data_source=None)
        self.ds = dataset
        self.n_way = int(n_way)
        self.k_shot = int(k_shot)
        self.q_query = int(max(0, q_query))
        self.episodes_per_epoch = int(episodes_per_epoch)
        self.rng = np.random.RandomState(int(rng_seed))
        self.sampling_mode = sampling_mode
        self.inject_unknown_ratio = float(inject_unknown_ratio)

        # ---------- Build class_to_indices if missing ----------
        if not hasattr(self.ds, "class_to_indices"):
            labels: List[int] = []

            if hasattr(self.ds, "df") and (self.ds.df is not None) and ("y" in self.ds.df.columns):
                labels = self.ds.df["y"].astype(int).tolist()
            else:
                for i in range(len(self.ds)):
                    item = self.ds[i]
                    if isinstance(item, dict) and ("y" in item):
                        y = item["y"]
                    elif isinstance(item, (list, tuple)) and len(item) >= 1:
                        y = item[-1]
                    else:
                        raise RuntimeError("FewShotEpisodeSampler: cannot infer labels.")
                    if isinstance(y, torch.Tensor):
                        y = int(y.item())
                    else:
                        y = int(y)
                    labels.append(y)

            c2i: Dict[int, List[int]] = defaultdict(list)
            for idx, yy in enumerate(labels):
                c2i[int(yy)].append(idx)
            self.ds.class_to_indices = dict(c2i)

            if not hasattr(self.ds, "num_classes"):
                self.ds.num_classes = (max(c2i.keys()) + 1) if len(c2i) else 0

        # ---------- Filter valid classes that can support k + q ----------
        need = self.k_shot + max(0, self.q_query)
        self.valid_classes = [c for c, idxs in self.ds.class_to_indices.items() if len(idxs) >= need]
        if len(self.valid_classes) < self.n_way:
            raise ValueError(f"Not enough classes with >= {need} samples. Have {len(self.valid_classes)}, need {self.n_way}.")

        # ---------- Buckets for head/mid/tail (optional) ----------
        counts = {int(c): len(self.ds.class_to_indices[c]) for c in self.valid_classes}
        vals = np.array(list(counts.values()), dtype=float)
        if len(vals) >= 3 and not np.allclose(vals.min(), vals.max()):
            q1 = np.quantile(vals, 1/3)
            q2 = np.quantile(vals, 2/3)
            self.buckets = {}
            for c, v in counts.items():
                if v <= q1:
                    self.buckets[int(c)] = "tail"
                elif v <= q2:
                    self.buckets[int(c)] = "mid"
                else:
                    self.buckets[int(c)] = "head"
        else:
            self.buckets = {c: "head" for c in counts.keys()}

        self.tail_classes = [c for c in self.valid_classes if self.buckets.get(c) == "tail"]
        self.head_classes = [c for c in self.valid_classes if self.buckets.get(c) == "head"]

        # ---------- Unknown pool (optional, for open-set) ----------
        if hasattr(self.ds, "df") and ("is_unknown" in getattr(self.ds, "df", {}).columns):
            self.ds_unknown_idxs = self.ds.df.index[self.ds.df["is_unknown"].values].tolist()
        else:
            self.ds_unknown_idxs = []

    def _sample_classes(self) -> np.ndarray:
        # "tail_mix": ensure some rare classes
        if self.sampling_mode == "tail_mix" and self.tail_classes and len(self.tail_classes) >= 2:
            k_tail = min(2, self.n_way)
            tails = self.rng.choice(self.tail_classes, k_tail, replace=False)
            pool  = [c for c in self.valid_classes if c not in tails]
            rest  = self.rng.choice(pool, self.n_way - k_tail, replace=False)
            return np.concatenate([tails, rest])

        # "real": proportional to frequency
        if self.sampling_mode == "real":
            weights = np.array([len(self.ds.class_to_indices[c]) for c in self.valid_classes], dtype=float)
            weights /= weights.sum()
            return self.rng.choice(self.valid_classes, self.n_way, replace=False, p=weights)

        # "balanced" default
        return self.rng.choice(self.valid_classes, self.n_way, replace=False)

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            classes = self._sample_classes()
            episode_indices: List[int] = []
            for c in classes:
                pool = self.ds.class_to_indices[c]
                chosen = self.rng.choice(pool, self.k_shot + self.q_query, replace=False)
                episode_indices.extend(chosen.tolist())

            # Inject unknowns into query (optional)
            if self.inject_unknown_ratio > 0 and self.ds_unknown_idxs and self.q_query > 0:
                q_total = self.n_way * self.q_query
                n_unk = max(1, int(self.inject_unknown_ratio * q_total))
                n_unk = min(n_unk, len(self.ds_unknown_idxs))
                if n_unk > 0:
                    unk_idxs = self.rng.choice(self.ds_unknown_idxs, n_unk, replace=False).tolist()
                    episode_indices.extend(unk_idxs)

            yield episode_indices

    def __len__(self) -> int:
        return self.episodes_per_epoch
