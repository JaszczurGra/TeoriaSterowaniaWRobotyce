import sys
from pathlib import Path

import numpy as np
import torch

import config as C

sys.path.insert(0, str(C.ROOT / "dataset"))
from dataset_reader_sens import F1tenthSensDataset 

V_SLICE = slice(3, 6)


def _list_npz():
    paths = sorted(C.DATASET_DIR.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"Brak plikow .npz w {C.DATASET_DIR}")
    return paths


def build_pooled(split="train", rollout_length=C.ROLLOUT_LENGTH, skip=3, device="cpu"):
    ds = F1tenthSensDataset(
        _list_npz(),
        rollout_length=rollout_length,
        skip=skip,
        sensitivities_strategy="ones",
        states_loss_strategy="v_only",
        split=split,
        split_val_tws=C.VAL_TWS,
    )

    ep_idx = ds.idxs[:, 0]
    t0 = ds.idxs[:, 1]
    seq = t0[:, None] + np.arange(rollout_length + 1)

    XS = ds.x[ep_idx[:, None], seq, :][:, :, V_SLICE]
    U = ds.u[ep_idx[:, None], seq, :]
    X0 = XS[:, 0, :]
    TW = ds.episode_tws[ep_idx]

    out = dict(
        X0=torch.as_tensor(X0, dtype=torch.float32, device=device),
        U=torch.as_tensor(U, dtype=torch.float32, device=device),
        XS=torch.as_tensor(XS, dtype=torch.float32, device=device),
        TW=torch.as_tensor(TW, dtype=torch.float32, device=device),
        x_std=torch.as_tensor(ds.x_std[V_SLICE], dtype=torch.float32, device=device),
        dx_dt_std=torch.as_tensor(ds.dx_dt_std[V_SLICE], dtype=torch.float32, device=device),
        episode_tws=ds.episode_tws,
        full_x=ds.x,
        full_u=ds.u,
    )
    print(f"[{split}] M={out['X0'].shape[0]} okien | tw={sorted(set(np.round(ds.episode_tws,3)))} "
          f"| x_std[vx,vy,r]={np.round(ds.x_std[V_SLICE],4)}")
    return out


def iter_minibatches(pool, batch_size, shuffle=True, seed=0, sub_len=None):
    M = pool["X0"].shape[0]
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(M, generator=g) if shuffle else torch.arange(M)
    for i in range(0, M, batch_size):
        idx = order[i:i + batch_size]
        U = pool["U"][idx]
        XS = pool["XS"][idx]
        if sub_len is not None:
            U = U[:, :sub_len + 1, :]
            XS = XS[:, :sub_len + 1, :]
        yield dict(X0=pool["X0"][idx], U=U, XS=XS, TW=pool["TW"][idx])


if __name__ == "__main__":
    tr = build_pooled("train")
    va = build_pooled("val")
    print("train X0", tr["X0"].shape, "U", tr["U"].shape, "XS", tr["XS"].shape, "TW", tr["TW"].shape)
    print("val   X0", va["X0"].shape)
    print("TW unique train:", torch.unique(tr["TW"]))
    print("TW unique val:", torch.unique(va["TW"]))
