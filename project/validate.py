import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from vehicle_model import VehicleModel
from data import build_pooled
from identify import evaluate

STATES = ["vx", "vy", "r"]

MODELS = [
    ("baseline", "params_baseline.pt"),
    ("control",  "params_control.pt"),
    ("mu2_only", "params_lt.pt"),
    ("joint_lt", "params_lt_joint.pt"),
    ("bfz",      "params_bfz.pt"),
]


def load_models():
    out = {}
    for name, fn in MODELS:
        p = C.RESULTS_DIR / fn
        if not p.exists():
            continue
        ck = torch.load(p, weights_only=False)
        m = VehicleModel(load_transfer=ck.get("load_transfer", False),
                         b_load_sens=ck.get("b_load_sens", False))
        m.load_state_dict(ck["state_dict"], strict=False)
        m.eval()
        out[name] = (m, ck)
    return out


def rmse_per_tw(model, pool):
    out = {}
    tw = pool["TW"].numpy()
    with torch.no_grad():
        pred = model.rollout(pool["X0"], pool["U"], pool["TW"])
        err = (pred - pool["XS"]).numpy()
    for t in sorted(set(np.round(tw, 3))):
        mask = np.round(tw, 3) == t
        out[t] = np.sqrt((err[mask] ** 2).mean(axis=(0, 1)))
    return out


def print_table(models, train, val):
    print("\n================  METRYKI (pelny rollout 40 krokow ~1.2 s)  ================")
    print(f"{'model':<10} | {'mu2(f,r,x)':>20} | {'VAL R2 [vx,vy,r] (tw=0.35)':>28} | {'TRAIN R2':>20}")
    print("-" * 92)
    for name, (m, ck) in models.items():
        rt, r2t = evaluate(m, train)
        rv, r2v = evaluate(m, val)
        pr = ck["params"]
        mu2 = f"{pr['f_mu2']:+.3f},{pr['r_mu2']:+.3f},{pr['x_mu2']:+.3f}"
        print(f"{name:<10} | {mu2:>20} | {str(np.round(r2v,3)):>28} | {str(np.round(r2t,3)):>20}")
    print("-" * 92)
    if "mu2_only" in models and "joint_lt" in models:
        a = models["mu2_only"][1]["params"]; b = models["joint_lt"][1]["params"]
        print(f"SYGNATURA NIEIDENTYFIKOWALNOSCI - zmiana znaku mu2:")
        print(f"  mu2_only (baza zamrozona): f={a['f_mu2']:+.3f}  (fizyczne, <0)")
        print(f"  joint_lt (baza wolna):     f={b['f_mu2']:+.3f}  (niefizyczne, >0)")
        print("  => mu2 nie jest identyfikowalny (znak zalezy od reszty parametrow).")


def fig_rollout(models, val, n=3):
    base = models["baseline"][0]
    lt = models.get("joint_lt", (None,))[0]
    g = torch.Generator().manual_seed(1)
    idx = torch.randperm(val["X0"].shape[0], generator=g)[:n]
    with torch.no_grad():
        pb = base.rollout(val["X0"][idx], val["U"][idx], val["TW"][idx]).numpy()
        pl = lt.rollout(val["X0"][idx], val["U"][idx], val["TW"][idx]).numpy() if lt else None
    meas = val["XS"][idx].numpy()
    t = np.arange(meas.shape[1]) * C.DT
    fig, ax = plt.subplots(n, 3, figsize=(13, 3 * n), squeeze=False)
    for i in range(n):
        for j in range(3):
            a = ax[i][j]
            a.plot(t, meas[i, :, j], "k-", lw=2, label="pomiar")
            a.plot(t, pb[i, :, j], "--", color="tab:red", label="baseline (jednoslad)")
            if pl is not None:
                a.plot(t, pl[i, :, j], ":", color="tab:blue", label="dwuslad+LT")
            a.set_title(f"okno {int(idx[i])} | {STATES[j]} | tw={val['TW'][idx[i]]:.2f}")
            a.grid(alpha=0.3)
            if i == 0 and j == 0:
                a.legend(fontsize=8)
            if i == n - 1:
                a.set_xlabel("czas [s]")
    fig.suptitle("Rollout otwartej petli na trzymanym tw=0.35 (pomiar vs model)")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "fig_rollout.png", dpi=110)
    plt.close(fig)


def fig_tire_curves(model):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    alpha = torch.linspace(-0.45, 0.45, 200)
    kappa = torch.linspace(-0.4, 0.4, 200)
    with torch.no_grad():
        for axi, (title, fn, Fz0, slip, xlab) in enumerate([
            ("Przod: Fy(alpha)", model.lat_front, C.FZ_F_STATIC, alpha, "alpha [rad]"),
            ("Tyl: Fy(alpha)",   model.lat_rear,  C.FZ_R_STATIC, alpha, "alpha [rad]"),
            ("Tyl: Fx(kappa)",   model.lon_rear,  C.FZ_R_STATIC, kappa, "kappa [-]"),
        ]):
            for sc, col in [(0.5, "tab:green"), (1.0, "tab:orange"), (1.5, "tab:red")]:
                Fz = torch.full_like(slip, Fz0 * sc)
                ax[axi].plot(slip.numpy(), fn(slip, Fz).numpy(), color=col, label=f"Fz={sc:.1f}·Fz0")
            ax[axi].set_title(title); ax[axi].axhline(0, color="k", lw=0.5); ax[axi].axvline(0, color="k", lw=0.5)
            ax[axi].grid(alpha=0.3); ax[axi].legend(fontsize=8)
            ax[axi].set_xlabel(xlab); ax[axi].set_ylabel("sila [N]")
    fig.suptitle("Zidentyfikowane krzywe opon Pacejka (model bazowy) - sila rosnie z obciazeniem Fz")
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "fig_tire_curves.png", dpi=110)
    plt.close(fig)


def fig_load_transfer(model, val):
    model.load_transfer = True
    X0 = val["X0"]; a_y = (X0[:, 0] * X0[:, 2]).abs()
    i = int(torch.argmax(a_y))
    U = val["U"][i:i + 1]; tw = val["TW"][i:i + 1]; x0 = X0[i:i + 1]
    with torch.no_grad():
        traj = model.rollout(x0, U, tw)[0]
        Fz = {k: [] for k in ["Fz_fl", "Fz_fr", "Fz_rl", "Fz_rr"]}; dlat = []
        for k in range(traj.shape[0]):
            _, _, _, info = model.forces(traj[k:k + 1], U[:, k, :], tw, detailed=True)
            for key in Fz:
                Fz[key].append(float(info[key]))
            dlat.append(float(info["dFz_lat_f"]))
    t = np.arange(traj.shape[0]) * C.DT
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for key, lab in [("Fz_fl", "przod-lewe"), ("Fz_fr", "przod-prawe"),
                     ("Fz_rl", "tyl-lewe"), ("Fz_rr", "tyl-prawe")]:
        ax[0].plot(t, Fz[key], label=f"Fz {lab}")
    ax[0].axhline(C.FZ_F_STATIC, ls=":", color="gray"); ax[0].axhline(C.FZ_R_STATIC, ls=":", color="gray")
    ax[0].set_title(f"Obciazenia kol w czasie (tw={float(tw):.2f})"); ax[0].set_xlabel("czas [s]")
    ax[0].set_ylabel("Fz [N]"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].plot(t, dlat, color="tab:purple")
    ax[1].set_title("Boczny transfer masy przedniej osi dFz_lat_f"); ax[1].set_xlabel("czas [s]")
    ax[1].set_ylabel("dFz [N]"); ax[1].grid(alpha=0.3)
    fig.suptitle("Mechanizm transferu masy reprezentowany przez model (model dwusladowy)")
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "fig_load_transfer.png", dpi=110)
    plt.close(fig)


def fig_ablation(models, val):
    order = [n for n in ["baseline", "control", "joint_lt"] if n in models]
    colors = {"baseline": "tab:gray", "control": "tab:red", "joint_lt": "tab:blue"}
    labels = {"baseline": "baseline (LT off)", "control": "kontrola (LT off, +trening)",
              "joint_lt": "dwuslad+LT (+trening)"}
    rmse = {n: list(rmse_per_tw(models[n][0], val).values())[0] for n in order}
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.6))
    x = np.arange(3); w = 0.8 / len(order)
    for k, n in enumerate(order):
        ax.bar(x + (k - (len(order) - 1) / 2) * w, rmse[n], w, label=labels[n], color=colors[n])
    ax.set_xticks(x); ax.set_xticklabels(STATES)
    ax.set_ylabel("RMSE (tw=0.35, trzymane)"); ax.set_title(
        "Ablacja dopasowana: transfer masy vs sama dluzsza optymalizacja")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "fig_ablation.png", dpi=110)
    plt.close(fig)


def main():
    train = build_pooled("train", skip=3)
    val = build_pooled("val", skip=3)
    models = load_models()
    print("Wczytane modele:", list(models))
    print_table(models, train, val)

    fig_rollout(models, val)
    base_m = models["baseline"][0]
    fig_tire_curves(base_m)
    fig_load_transfer(base_m, val)
    fig_ablation(models, val)
    print(f"\nWykresy zapisane w {C.FIGURES_DIR}")


if __name__ == "__main__":
    main()
