import torch
import torch.nn as nn

import config as C


def _inv_softplus(y):
    y = torch.as_tensor(float(y))
    return torch.log(torch.expm1(y.clamp(min=1e-6)))


class VehicleModel(nn.Module):
    def __init__(self, load_transfer=True, combined_slip=True, n_lt_iters=3, b_load_sens=False):
        super().__init__()
        self.load_transfer = load_transfer
        self.combined_slip = combined_slip
        self.n_lt_iters = n_lt_iters

        self.register_buffer("m", torch.tensor(C.M))
        self.register_buffer("Iz", torch.tensor(C.IZ))
        self.register_buffer("lf", torch.tensor(C.LF))
        self.register_buffer("lr", torch.tensor(C.LR))
        self.register_buffer("L", torch.tensor(C.L))
        self.register_buffer("h", torch.tensor(C.H_CG))
        self.register_buffer("Fz_f0", torch.tensor(C.FZ_F_STATIC))
        self.register_buffer("Fz_r0", torch.tensor(C.FZ_R_STATIC))
        self.eps = C.EPS

        lat, lon, dt = C.PACEJKA_LAT_INIT, C.PACEJKA_LON_INIT, C.DRIVETRAIN_INIT
        p = {}
        p["f_B"] = _inv_softplus(lat["B"]);   p["f_C"] = _inv_softplus(lat["C"])
        p["f_mu0"] = _inv_softplus(lat["mu0"]); p["f_mu2"] = torch.tensor(float(lat["mu2"]))
        p["f_E"] = torch.tensor(float(lat["E"]))
        p["r_B"] = _inv_softplus(lat["B"]);   p["r_C"] = _inv_softplus(lat["C"])
        p["r_mu0"] = _inv_softplus(lat["mu0"]); p["r_mu2"] = torch.tensor(float(lat["mu2"]))
        p["r_E"] = torch.tensor(float(lat["E"]))
        p["x_B"] = _inv_softplus(lon["Bx"]);  p["x_C"] = _inv_softplus(lon["Cx"])
        p["x_mu0"] = _inv_softplus(lon["mux0"]); p["x_mu2"] = torch.tensor(float(lon["mux2"]))
        p["x_E"] = torch.tensor(float(lon["Ex"]))
        p["d_Cr0"] = _inv_softplus(dt["Cr0"]); p["d_Crv"] = _inv_softplus(dt["Cr_v"])
        p["d_Cd"] = _inv_softplus(dt["Cd"])
        p["f_B1"] = torch.tensor(0.0); p["r_B1"] = torch.tensor(0.0); p["x_B1"] = torch.tensor(0.0)
        self.raw = nn.ParameterDict({k: nn.Parameter(v.clone().float()) for k, v in p.items()})

        self._pos = {"f_B", "f_C", "f_mu0", "r_B", "r_C", "r_mu0",
                     "x_B", "x_C", "x_mu0", "d_Cr0", "d_Crv", "d_Cd"}
        self.b_load_sens = b_load_sens

        self.groups = {
            "lon": ["x_B", "x_C", "x_mu0", "x_E", "d_Cr0", "d_Crv", "d_Cd"],
            "lat": ["f_B", "f_C", "f_mu0", "f_E", "r_B", "r_C", "r_mu0", "r_E"],
            "mu2": ["f_mu2", "r_mu2", "x_mu2"],
            "b1": ["f_B1", "r_B1", "x_B1"],
        }

    def g(self, name):
        v = self.raw[name]
        return torch.nn.functional.softplus(v) if name in self._pos else v

    def params_readable(self):
        return {k: float(self.g(k)) for k in self.raw.keys()}

    def set_stage(self, stage):
        for prm in self.raw.values():
            prm.requires_grad_(False)
        if stage == "lon":
            train = self.groups["lon"]; self.load_transfer = False
        elif stage == "lat":
            train = self.groups["lat"]; self.load_transfer = False
        elif stage == "joint":
            train = self.groups["lon"] + self.groups["lat"]; self.load_transfer = False
        elif stage == "mu2_only":
            train = self.groups["mu2"]
            self.load_transfer = True; self.b_load_sens = False
        elif stage == "joint_lt":
            train = self.groups["lon"] + self.groups["lat"] + self.groups["mu2"]
            self.load_transfer = True; self.b_load_sens = False
        elif stage == "bfz_only":
            train = self.groups["b1"]
            self.load_transfer = True; self.b_load_sens = True
        elif stage == "bfz_mu2":
            train = self.groups["b1"] + self.groups["mu2"]
            self.load_transfer = True; self.b_load_sens = True
        else:
            raise ValueError(stage)
        for name in train:
            self.raw[name].requires_grad_(True)
        return [self.raw[name] for name in train]

    def _mu(self, Fz, mu0, mu2, Fz0):
        return (mu0 + mu2 * (Fz - Fz0)).clamp(min=0.05)

    def _pacejka(self, slip, Fz, B, C_, mu0, mu2, E, Fz0):
        D = self._mu(Fz, mu0, mu2, Fz0) * Fz
        Bs = B * slip
        return D * torch.sin(C_ * torch.atan(Bs - E * (Bs - torch.atan(Bs))))

    def _pacejka_peak(self, Fz, mu0, mu2, Fz0):
        return self._mu(Fz, mu0, mu2, Fz0) * Fz

    def _B_eff(self, prefix, Fz, Fz0):
        B0 = self.g(prefix + "_B")
        if self.b_load_sens:
            return (B0 + self.raw[prefix + "_B1"] * (Fz - Fz0)).clamp(min=0.5)
        return B0

    def lat_front(self, alpha, Fz):
        return self._pacejka(alpha, Fz, self._B_eff("f", Fz, self.Fz_f0), self.g("f_C"),
                             self.g("f_mu0"), self.g("f_mu2"), self.g("f_E"), self.Fz_f0)

    def lat_rear(self, alpha, Fz):
        return self._pacejka(alpha, Fz, self._B_eff("r", Fz, self.Fz_r0), self.g("r_C"),
                             self.g("r_mu0"), self.g("r_mu2"), self.g("r_E"), self.Fz_r0)

    def lon_rear(self, kappa, Fz):
        return self._pacejka(kappa, Fz, self._B_eff("x", Fz, self.Fz_r0), self.g("x_C"),
                             self.g("x_mu0"), self.g("x_mu2"), self.g("x_E"), self.Fz_r0)

    def _rear_combined(self, kappa, alpha, Fz):
        Fx0 = self.lon_rear(kappa, Fz)
        Fy0 = self.lat_rear(alpha, Fz)
        if not self.combined_slip:
            return Fx0, Fy0
        Dx = self._pacejka_peak(Fz, self.g("x_mu0"), self.g("x_mu2"), self.Fz_r0)
        Dy = self._pacejka_peak(Fz, self.g("r_mu0"), self.g("r_mu2"), self.Fz_r0)
        rho = torch.sqrt((Fx0 / (Dx + 1e-6)) ** 2 + (Fy0 / (Dy + 1e-6)) ** 2 + 1e-12)
        scale = 1.0 / rho.clamp(min=1.0)
        return Fx0 * scale, Fy0 * scale

    def forces(self, s, u, tw, detailed=False):
        vx, vy, r = s[..., 0], s[..., 1], s[..., 2]
        ws, delta = u[..., 0], u[..., 1]
        tw = torch.as_tensor(tw, dtype=vx.dtype, device=vx.device)

        vx_safe = vx.clamp(min=self.eps)
        alpha_f = delta - torch.atan((vy + self.lf * r) / vx_safe)
        alpha_r = -torch.atan((vy - self.lr * r) / vx_safe)
        kappa_r = (ws - vx) / vx_safe
        cd, sd = torch.cos(delta), torch.sin(delta)
        F_resist = self.g("d_Cr0") + self.g("d_Crv") * vx + self.g("d_Cd") * vx * vx

        Fz_fl = Fz_fr = self.Fz_f0 * torch.ones_like(vx)
        Fz_rl = Fz_rr = self.Fz_r0 * torch.ones_like(vx)
        n_iter = self.n_lt_iters if self.load_transfer else 1
        for _ in range(n_iter):
            Fy_fl = self.lat_front(alpha_f, Fz_fl)
            Fy_fr = self.lat_front(alpha_f, Fz_fr)
            Fx_rl, Fy_rl = self._rear_combined(kappa_r, alpha_r, Fz_rl)
            Fx_rr, Fy_rr = self._rear_combined(kappa_r, alpha_r, Fz_rr)

            Fy_f_axle = Fy_fl + Fy_fr
            Fy_r_axle = Fy_rl + Fy_rr
            Fx_r_axle = Fx_rl + Fx_rr

            Fy_f_body = Fy_f_axle * cd
            Fx_sum = Fx_r_axle - Fy_f_axle * sd - F_resist
            Fy_sum = Fy_f_body + Fy_r_axle

            if not self.load_transfer:
                break
            dFz_long = Fx_sum * self.h / self.L
            dFz_lat_f = Fy_f_body * self.h / tw
            dFz_lat_r = Fy_r_axle * self.h / tw
            half_long = 0.5 * dFz_long
            Fz_fl = (self.Fz_f0 - half_long - dFz_lat_f).clamp(min=1e-3)
            Fz_fr = (self.Fz_f0 - half_long + dFz_lat_f).clamp(min=1e-3)
            Fz_rl = (self.Fz_r0 + half_long - dFz_lat_r).clamp(min=1e-3)
            Fz_rr = (self.Fz_r0 + half_long + dFz_lat_r).clamp(min=1e-3)

        Mz = self.lf * Fy_f_body - self.lr * Fy_r_axle + 0.5 * tw * (Fx_rr - Fx_rl)

        if detailed:
            info = dict(Fz_fl=Fz_fl, Fz_fr=Fz_fr, Fz_rl=Fz_rl, Fz_rr=Fz_rr,
                        Fy_f=Fy_f_axle, Fy_r=Fy_r_axle, Fx_r=Fx_r_axle,
                        alpha_f=alpha_f, alpha_r=alpha_r, kappa_r=kappa_r,
                        dFz_lat_f=(Fy_f_body * self.h / tw) if self.load_transfer else torch.zeros_like(vx))
            return Fx_sum, Fy_sum, Mz, info
        return Fx_sum, Fy_sum, Mz

    def dynamics(self, s, u, tw):
        vx, vy, r = s[..., 0], s[..., 1], s[..., 2]
        Fx, Fy, Mz = self.forces(s, u, tw)
        dvx = Fx / self.m + vy * r
        dvy = Fy / self.m - vx * r
        dr = Mz / self.Iz
        return torch.stack([dvx, dvy, dr], dim=-1)

    def rk4_step(self, s, u, tw, dt):
        k1 = self.dynamics(s, u, tw)
        k2 = self.dynamics(s + 0.5 * dt * k1, u, tw)
        k3 = self.dynamics(s + 0.5 * dt * k2, u, tw)
        k4 = self.dynamics(s + dt * k3, u, tw)
        return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rollout(self, x0, u_seq, tw, dt=C.DT):
        T1 = u_seq.shape[1]
        s = x0
        out = [s]
        for k in range(T1 - 1):
            s = self.rk4_step(s, u_seq[:, k, :], tw, dt)
            out.append(s)
        return torch.stack(out, dim=1)
