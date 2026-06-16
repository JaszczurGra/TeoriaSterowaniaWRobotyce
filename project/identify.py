import argparse
import numpy as np
import torch

import config as C
from vehicle_model import VehicleModel
from data import build_pooled, iter_minibatches


def weighted_loss(pred, meas, x_std, chan_w):
    w = (chan_w / x_std).to(pred.dtype)
    err2 = ((pred - meas) * w) ** 2
    return err2.mean()


def evaluate(model, pool, sub_len=None):
    model.eval()
    with torch.no_grad():
        U, XS = pool["U"], pool["XS"]
        if sub_len is not None:
            U, XS = U[:, :sub_len + 1], XS[:, :sub_len + 1]
        pred = model.rollout(pool["X0"], U, pool["TW"])
        err = pred - XS
        rmse = torch.sqrt((err ** 2).mean(dim=(0, 1)))
        ss_res = (err ** 2).sum(dim=(0, 1))
        ss_tot = ((XS - XS.mean(dim=(0, 1), keepdim=True)) ** 2).sum(dim=(0, 1))
        r2 = 1.0 - ss_res / ss_tot
    return rmse.cpu().numpy(), r2.cpu().numpy()


def train_stage(model, pool, stage, *, epochs, lr, batch_size, sub_len, chan_w,
                clip=10.0, seed=0, log_every=10):
    params = model.set_stage(stage)
    model.train()
    opt = torch.optim.Adam(params, lr=lr)
    x_std = pool["x_std"]
    chan_w = torch.tensor(chan_w, dtype=torch.float32)
    print(f"\n=== etap '{stage}' | LT={model.load_transfer} | "
          f"{sum(p.numel() for p in params)} param. | sub_len={sub_len} lr={lr} ===")
    for ep in range(epochs):
        tot = 0.0; nb = 0
        for batch in iter_minibatches(pool, batch_size, shuffle=True, seed=seed + ep, sub_len=sub_len):
            opt.zero_grad()
            pred = model.rollout(batch["X0"], batch["U"], batch["TW"])
            loss = weighted_loss(pred, batch["XS"], x_std, chan_w)
            if not torch.isfinite(loss):
                print("  !! niefinitna strata - pomijam batch"); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            tot += float(loss); nb += 1
        if ep % log_every == 0 or ep == epochs - 1:
            rmse, r2 = evaluate(model, pool, sub_len=sub_len)
            print(f"  ep {ep:3d}  loss={tot/max(nb,1):.4f}  "
                  f"RMSE[vx,vy,r]={np.round(rmse,4)}  R2={np.round(r2,3)}")
    return model


def report(model, train, val, tag=""):
    rmse_t, r2_t = evaluate(model, train)
    rmse_v, r2_v = evaluate(model, val)
    print(f"\n----- WYNIK {tag} (pełny rollout {C.ROLLOUT_LENGTH} kroków) -----")
    print(f"  TRAIN RMSE[vx,vy,r]={np.round(rmse_t,4)}  R2={np.round(r2_t,3)}")
    print(f"  VAL   RMSE[vx,vy,r]={np.round(rmse_v,4)}  R2={np.round(r2_v,3)}  (tw=0.35 held-out)")
    return dict(rmse_train=rmse_t, r2_train=r2_t, rmse_val=rmse_v, r2_val=r2_v)


def fit_baseline(train, val, seed=C.SEED):
    torch.manual_seed(seed)
    model = VehicleModel(load_transfer=False)
    train_stage(model, train, "lon",   epochs=40, lr=0.05, batch_size=512, sub_len=8,  chan_w=[1, 0, 0], seed=seed)
    train_stage(model, train, "lat",   epochs=50, lr=0.05, batch_size=512, sub_len=10, chan_w=[0, 1, 1], seed=seed)
    train_stage(model, train, "joint", epochs=60, lr=0.02, batch_size=512, sub_len=20, chan_w=[1, 1, 1], seed=seed)
    metrics = report(model, train, val, tag="BASELINE (bez transferu masy)")
    torch.save({"state_dict": model.state_dict(),
                "params": model.params_readable(),
                "metrics": metrics, "load_transfer": False},
               C.RESULTS_DIR / "params_baseline.pt")
    print(f"\nZapisano -> {C.RESULTS_DIR / 'params_baseline.pt'}")
    return model


def _save(model, metrics, path):
    torch.save({"state_dict": model.state_dict(),
                "params": model.params_readable(),
                "metrics": metrics,
                "load_transfer": model.load_transfer,
                "b_load_sens": model.b_load_sens}, path)
    print(f"Zapisano -> {path}")


def fit_load_transfer(train, val, seed=C.SEED):
    base = torch.load(C.RESULTS_DIR / "params_baseline.pt", weights_only=False)

    m_mu2 = VehicleModel(load_transfer=True)
    m_mu2.load_state_dict(base["state_dict"], strict=False)
    print("\n### Czysta ablacja: dofit TYLKO mu2 (reszta = baseline) ###")
    train_stage(m_mu2, train, "mu2_only", epochs=60, lr=0.01, batch_size=512, sub_len=25,
                chan_w=[1, 1, 1], seed=seed)
    met_mu2 = report(m_mu2, train, val, tag="TRANSFER MASY (tylko mu2)")
    pr = m_mu2.params_readable()
    print("mu2 (f,r,x):", round(pr["f_mu2"], 4), round(pr["r_mu2"], 4), round(pr["x_mu2"], 4))
    _save(m_mu2, met_mu2, C.RESULTS_DIR / "params_lt.pt")

    m_joint = VehicleModel(load_transfer=True)
    m_joint.load_state_dict(base["state_dict"], strict=False)
    print("\n### Pelny dofit z transferem masy (joint_lt) ###")
    train_stage(m_joint, train, "joint_lt", epochs=80, lr=0.01, batch_size=512, sub_len=25,
                chan_w=[1, 1, 1], seed=seed)
    met_joint = report(m_joint, train, val, tag="TRANSFER MASY (pelny joint)")
    prj = m_joint.params_readable()
    print("mu2 (f,r,x):", round(prj["f_mu2"], 4), round(prj["r_mu2"], 4), round(prj["x_mu2"], 4))
    _save(m_joint, met_joint, C.RESULTS_DIR / "params_lt_joint.pt")
    return m_mu2, m_joint


def fit_control(train, val, seed=C.SEED):
    base = torch.load(C.RESULTS_DIR / "params_baseline.pt", weights_only=False)
    m = VehicleModel(load_transfer=False)
    m.load_state_dict(base["state_dict"], strict=False)
    print("\n### Kontrola: kontynuacja baseline (LT OFF), harmonogram = joint_lt ###")
    train_stage(m, train, "joint", epochs=80, lr=0.01, batch_size=512, sub_len=25,
                chan_w=[1, 1, 1], seed=seed)
    met = report(m, train, val, tag="KONTROLA (LT OFF, dluzszy trening)")
    _save(m, met, C.RESULTS_DIR / "params_control.pt")
    return m


def fit_bfz(train, val, seed=C.SEED):
    base = torch.load(C.RESULTS_DIR / "params_baseline.pt", weights_only=False)
    m = VehicleModel(load_transfer=True, b_load_sens=True)
    m.load_state_dict(base["state_dict"], strict=False)
    print("\n### Ablacja B(Fz): czulosc sztywnosci na obciazenie (mechanizm 1. rzedu) ###")
    train_stage(m, train, "bfz_only", epochs=60, lr=0.01, batch_size=512, sub_len=25,
                chan_w=[1, 1, 1], seed=seed)
    met = report(m, train, val, tag="TRANSFER MASY B(Fz)")
    pr = m.params_readable()
    print("B1 (f,r,x):", round(pr["f_B1"], 4), round(pr["r_B1"], 4), round(pr["x_B1"], 4))
    _save(m, met, C.RESULTS_DIR / "params_bfz.pt")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-transfer", action="store_true", help="dofit z transferem masy (mu2)")
    ap.add_argument("--bfz", action="store_true", help="ablacja B(Fz) (mechanizm 1. rzedu)")
    ap.add_argument("--control", action="store_true", help="kontrola dopasowana (LT OFF, dluzszy trening)")
    ap.add_argument("--skip", type=int, default=3)
    args = ap.parse_args()

    train = build_pooled("train", skip=args.skip)
    val = build_pooled("val", skip=args.skip)

    if args.control:
        fit_control(train, val)
    elif args.bfz:
        fit_bfz(train, val)
    elif args.load_transfer:
        fit_load_transfer(train, val)
    else:
        fit_baseline(train, val)
