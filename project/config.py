from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset" / "f1tenth_long_track_sens"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

DT = 0.03

M = 3.74
IZ = 0.047
LF = 0.159
LR = 0.171
L = LF + LR
G = 9.81
H_CG = 0.04

FZ_F_STATIC = M * G * LR / L / 2.0
FZ_R_STATIC = M * G * LF / L / 2.0

EPS = 1e-3
VX_GATE = 0.64

VAL_TWS = [0.35]

ROLLOUT_LENGTH = 40

PACEJKA_LAT_INIT = dict(B=10.0, C=1.5, mu0=1.0, mu2=0.0, E=0.0)
PACEJKA_LON_INIT = dict(Bx=10.0, Cx=1.5, mux0=1.0, mux2=0.0, Ex=0.0)
DRIVETRAIN_INIT = dict(Cr0=0.1, Cr_v=0.05, Cd=0.01)

SEED = 42
