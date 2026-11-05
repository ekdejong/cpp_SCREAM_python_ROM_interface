"""
TorchScript-compatible AE-SINDy collision-coalescence ROM for SCREAM/EAMxx.

This is a pure-PyTorch rewrite of the Python side of
cpp_SCREAM_python_ROM_interface (py_interface.py + py_ROM.py + rom_utils.py):

    (Nc, Nr, qc, qr, mu_c, mu_r[, dt])        bulk P3 state, per volume of air
      -> gamma PSDs binned on 63 log-r bins   (closed form, regularized incomplete gamma)
      -> encoder -> latent z, append scaled total mass M
      -> integrate dz/dt = SINDy(z, M) over dt (Dormand-Prince 5(4) or fixed-step RK4)
      -> decoder -> PSD after dt
      -> split at the radius threshold into cloud / rain
      -> tendencies (dNr/dt, dNc/dt, dqc/dt, dqr/dt), with the same sign
         limiting as py_interface.py (no tendencies unless cloud mass decreases)

Everything in `forward` is TorchScript-scriptable, vectorized over any batch
shape, and free of numpy/scipy, so the exported module can be loaded from C++
with `torch::jit::load` (or from Fortran via FTorch).

Units (same contract as py_interface.ROM_interface):
    inputs : Nc, Nr [1/m^3],  qc, qr [kg/m^3]      (i.e. rho_air * mixing ratio)
             mu_c, mu_r [-]  gamma shape parameters (tensors, broadcastable)
             dt [s]          optional float; defaults to the dt set at export

Rain-only columns (qc < qsmall < qr) get dNr/dt only: the total number change over
all bins. dqc/dt = dqr/dt = dNc/dt = 0 there.
With limit_nr (default), dNr/dt can be positive only where dNc/dt < 0; elsewhere it
is capped at 0. In rain-only columns this means dNr/dt <= 0.
    outputs: dNr/dt, dNc/dt [1/m^3/s],  dqc/dt, dqr/dt [kg/m^3/s]
"""
import math
from itertools import combinations_with_replacement
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn


# ----------------------------------------------------------------------------
# Network components. Attribute names match rom_utils.py so the existing .pth
# state dicts load unchanged.
# ----------------------------------------------------------------------------
class FFNNEncoder(nn.Module):
    def __init__(self, n_bins: int = 63, n_latent: int = 3):
        super().__init__()
        self.layer1 = nn.Linear(n_bins, n_bins // 2)
        self.layer2 = nn.Linear(n_bins // 2, n_bins // 4)
        self.layer3 = nn.Linear(n_bins // 4, n_bins // 8)
        self.layer4 = nn.Linear(n_bins // 8, n_latent)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return self.layer4(x)


class FFNNDecoder(nn.Module):
    def __init__(self, n_bins: int = 63, n_latent: int = 3):
        super().__init__()
        self.layer1 = nn.Linear(n_latent, n_bins // 8)
        self.layer2 = nn.Linear(n_bins // 8, n_bins // 4)
        self.layer3 = nn.Linear(n_bins // 4, n_bins // 2)
        self.layer4 = nn.Linear(n_bins // 2, n_bins)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return torch.softmax(self.layer4(x), dim=-1)  # distribution=True


def _library_indices(n: int, poly_order: int) -> List[List[int]]:
    """Monomial index table in the same order as rom_utils.sindy_library_tensor.

    Each row lists the latent indices multiplied together; index `n` points at
    an appended column of ones and pads lower-order terms.
    """
    rows: List[List[int]] = []
    for order in range(poly_order + 1):
        for combo in combinations_with_replacement(range(n), order):
            rows.append(list(combo) + [n] * (poly_order - order))
    if poly_order == 0:
        rows = [[n]]
    return rows


class SINDyDeriv(nn.Module):
    def __init__(self, n_latent: int = 4, poly_order: int = 2):
        super().__init__()
        idx = _library_indices(n_latent, poly_order)
        self.n_latent = n_latent
        self.library_size = len(idx)
        self.width = max(poly_order, 1)
        self.sindy_coeffs = nn.Linear(self.library_size, n_latent, bias=False)
        # not in the .pth files -> non-persistent (still serialized by TorchScript)
        self.register_buffer(
            "lib_idx", torch.tensor(idx, dtype=torch.long).reshape(-1), persistent=False
        )

    def forward(self, z: Tensor) -> Tensor:
        b = z.shape[0]
        z_aug = torch.cat([z, torch.ones_like(z[:, :1])], dim=-1)
        lib = z_aug.index_select(1, self.lib_idx).reshape(b, self.library_size, self.width)
        return self.sindy_coeffs(lib.prod(dim=-1))


def infer_architecture(sd: dict) -> dict:
    """Read n_bins, n_latent and poly_order from an AE-SINDy state dict."""
    n_bins = int(sd["encoder.layer1.weight"].shape[1])
    n_latent = int(sd["encoder.layer4.weight"].shape[0])
    n_out, lib_size = (int(v) for v in sd["dzdt.sindy_coeffs.weight"].shape)
    if n_out != n_latent + 1:
        raise ValueError(f"SINDy has {n_out} states, expected n_latent + 1 = {n_latent + 1}")
    for order in range(0, 8):
        if len(_library_indices(n_out, order)) == lib_size:
            return {"n_bins": n_bins, "n_latent": n_latent, "poly_order": order}
    raise ValueError(f"no polynomial order gives a library of size {lib_size} for {n_out} states")


# ----------------------------------------------------------------------------
# Full interface
# ----------------------------------------------------------------------------
class SDMCollisionROM(nn.Module):
    """Bulk (Nc, Nr, qc, qr) -> collision-coalescence tendencies via AE-SINDy.

    Everything passed to the constructor is frozen into the exported
    TorchScript module, including `r_threshold` (cloud/rain split radius).
    """

    def __init__(
        self,
        weights: Union[str, dict],
        zlim: Union[str, np.ndarray],
        r_threshold: float = 50e-6,
        dt: float = 100.0,
        qsmall: float = 1e-18,
        integrator: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-6,
        max_steps: int = 1000,
        n_substeps: int = 10,
        n_bins: Optional[int] = None,
        r_min: float = 1e-6,
        r_max: float = 5e-3,
        rhow: float = 1000.0,
        m_scale: float = 0.01046717958952477,
        n_latent: Optional[int] = None,
        poly_order: Optional[int] = None,
        limit_nr: bool = True,
        legacy_dlnr: bool = True,
        override_mass_limits: bool = True,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if integrator not in ("dopri5", "rk4"):
            raise ValueError("integrator must be 'dopri5' or 'rk4'")

        # ---- networks: architecture read from the weights ------------------
        sd = torch.load(weights, map_location="cpu", weights_only=True) if isinstance(weights, str) else weights
        arch = infer_architecture(sd)
        for name, given in (("n_bins", n_bins), ("n_latent", n_latent), ("poly_order", poly_order)):
            if given is not None and given != arch[name]:
                raise ValueError(f"{name}={given} requested but the weights have {name}={arch[name]}")
        n_bins, n_latent, poly_order = arch["n_bins"], arch["n_latent"], arch["poly_order"]
        self.n_bins = n_bins
        self.n_latent = n_latent
        self.poly_order = poly_order
        self.encoder = FFNNEncoder(n_bins, n_latent)
        self.decoder = FFNNDecoder(n_bins, n_latent)
        self.dzdt = SINDyDeriv(n_latent + 1, poly_order)
        for name in ("encoder", "decoder", "dzdt"):
            sub = {k[len(name) + 1:]: v for k, v in sd.items() if k.startswith(name + ".")}
            getattr(self, name).load_state_dict(sub, strict=True)

        # ---- scalar config (frozen into the TorchScript module) -------------
        self.dt = float(dt)
        self.qsmall = float(qsmall)
        self.rhow = float(rhow)
        self.m_scale = float(m_scale)
        self.use_dopri5 = integrator == "dopri5"
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.max_steps = int(max_steps)
        self.n_substeps = int(n_substeps)
        # dNr/dt > 0 only where dNc/dt < 0
        self.limit_nr = bool(limit_nr)
        # Dormand-Prince 5(4) step controller constants (identical to scipy RK45)
        self.safety = 0.9
        self.min_factor = 0.2
        self.max_factor = 10.0

        # ---- bin grid ---------------------------------------------------------
        n_edges = n_bins + 1
        r_edges = np.exp(np.linspace(np.log(r_min), np.log(r_max), n_edges))
        # NOTE: py_interface.py divides by NUMEDG (=64), not NUMBIN (=63). Within
        # this pipeline the factor cancels in every tendency and only rescales the
        # latent mass coordinate M = sum(dm/dlnr)/M_SCALE, so keep it if the ROM
        # was trained with the same convention.
        self.dlnr = float((np.log(r_max) - np.log(r_min)) / (n_edges if legacy_dlnr else n_bins))
        m_to_n = (
            (1.0 / r_edges[:-1] ** 3 - 1.0 / r_edges[1:] ** 3)
            / np.log(r_edges[1:] / r_edges[:-1]) / (4.0 * np.pi * rhow)
        )
        # find_nearest(R_EDG, r_threshold): cloud = bins 0..cutoff_idx inclusive,
        # i.e. the split falls at edge R_EDG[cutoff_idx + 1] (as in the original).
        cutoff_idx = int(np.abs(r_edges - r_threshold).argmin())
        cloud_mask = (np.arange(n_bins) <= cutoff_idx).astype(np.float64)
        self.r_threshold = float(r_threshold)
        self.cutoff_idx = cutoff_idx
        self.r_split = float(r_edges[min(cutoff_idx + 1, n_bins)])

        # ---- latent bounds -------------------------------------------------
        zl = np.load(zlim)["zlim"] if isinstance(zlim, str) else np.asarray(zlim)
        zl = np.array(zl, dtype=np.float64, copy=True)
        if zl.shape != (n_latent + 1, 2):
            raise ValueError(f"zlim has shape {zl.shape}, expected {(n_latent + 1, 2)} for these weights")
        if override_mass_limits:
            zl[-1, 0], zl[-1, 1] = 0.0, np.inf

        self.register_buffer("d_edges", torch.tensor(2.0 * r_edges))        # diameters [m]
        self.register_buffer("m_to_n", torch.tensor(m_to_n))
        self.register_buffer("cloud_mask", torch.tensor(cloud_mask))
        self.register_buffer("z_lo", torch.tensor(zl[:, 0]))
        self.register_buffer("z_hi", torch.tensor(zl[:, 1]))
        self.to(dtype)
        self.requires_grad_(False)  # inference only; no autograd graph in C++
        self.eval()

    # ------------------------------------------------------------------------
    # Bulk -> bins (closed form of the scipy.quad integrals in py_interface)
    #   n_i = N * [P(mu+1, lam*D_{i+1}) - P(mu+1, lam*D_i)]
    #   m_i = q * [P(mu+4, lam*D_{i+1}) - P(mu+4, lam*D_i)]
    # with P the regularized lower incomplete gamma function.
    # ------------------------------------------------------------------------
    def _gamma_bin_fraction(self, a: Tensor, x: Tensor) -> Tensor:
        # a: (B, 1) shape parameter, x: (B, n_edges) = lambda * D_edges
        aa = a.expand_as(x)
        p = torch.special.gammainc(aa, x)
        q = torch.special.gammaincc(aa, x)
        lower = p[:, 1:] - p[:, :-1]
        upper = q[:, :-1] - q[:, 1:]  # better conditioned in the right tail
        return torch.clamp(torch.where(x[:, :-1] > a, upper, lower), min=0.0)

    def _bin_gamma(self, N: Tensor, q: Tensor, mu: Tensor, on: Tensor) -> Tuple[Tensor, Tensor]:
        present = on & (N > 0.0)
        Ns = torch.where(present, N, torch.ones_like(N))
        qs = torch.where(present, q, torch.ones_like(q))
        # Gamma(mu+4)/Gamma(mu+1) = (mu+1)(mu+2)(mu+3), exact for any real mu > -1
        gratio = (mu + 1.0) * (mu + 2.0) * (mu + 3.0)
        lam = torch.pow(math.pi / 6.0 * self.rhow * Ns * gratio / qs, 1.0 / 3.0)  # [1/m]
        x = lam.unsqueeze(1) * self.d_edges.unsqueeze(0)
        keep = present.to(N.dtype).unsqueeze(1) / self.dlnr
        mu1 = mu.unsqueeze(1)
        dndlnr = Ns.unsqueeze(1) * self._gamma_bin_fraction(mu1 + 1.0, x) * keep
        dmdlnr = qs.unsqueeze(1) * self._gamma_bin_fraction(mu1 + 4.0, x) * keep
        return dndlnr, dmdlnr

    # ------------------------------------------------------------------------
    # Latent dynamics: SINDy RHS with derivative zeroed outside zlim
    # (same as the clipping inside rom_utils.simulate)
    # ------------------------------------------------------------------------
    def _rhs(self, z: Tensor) -> Tensor:
        dz = self.dzdt(z)
        frozen = (z >= self.z_hi) | (z <= self.z_lo)
        return torch.where(frozen, torch.zeros_like(dz), dz)

    def _rms(self, x: Tensor) -> Tensor:
        return torch.sqrt(torch.mean(x * x, dim=-1))

    def _integrate_rk4(self, z: Tensor, T: float) -> Tensor:
        h = T / self.n_substeps
        for _ in range(self.n_substeps):
            k1 = self._rhs(z)
            k2 = self._rhs(z + 0.5 * h * k1)
            k3 = self._rhs(z + 0.5 * h * k2)
            k4 = self._rhs(z + h * k3)
            z = z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return z

    def _integrate_dopri5(self, z0: Tensor, T: float) -> Tensor:
        """Batched Dormand-Prince 5(4), step-size control as in scipy RK45.

        Every column gets its own step size; the loop runs until every column
        has reached t = dt (or max_steps iterations).
        """
        rtol = self.rtol
        atol = self.atol
        ex = -0.2  # -1/(error_estimator_order + 1)

        f0 = self._rhs(z0)
        # initial step (scipy select_initial_step, order 4)
        scale = atol + torch.abs(z0) * rtol
        d0 = self._rms(z0 / scale)
        d1 = self._rms(f0 / scale)
        h0 = torch.where((d0 < 1e-5) | (d1 < 1e-5), torch.full_like(d0, 1e-6), 0.01 * d0 / d1)
        h0 = torch.clamp(h0, max=T)
        f1 = self._rhs(z0 + h0.unsqueeze(1) * f0)
        d2 = self._rms((f1 - f0) / scale) / h0
        h1 = torch.where(
            (d1 <= 1e-15) & (d2 <= 1e-15),
            torch.clamp(h0 * 1e-3, min=1e-6),
            torch.pow(0.01 / torch.maximum(d1, d2), 0.2),
        )
        h_abs = torch.clamp(torch.minimum(100.0 * h0, h1), max=T)

        t = torch.zeros_like(d0)
        z = z0
        f = f0
        done = torch.zeros_like(d0, dtype=torch.bool)
        rejected = torch.zeros_like(done)
        it = 0
        while bool((~done).any()) and it < self.max_steps:
            it += 1
            active = ~done
            t_new = torch.clamp(t + h_abs, max=T)
            h_used = torch.where(active, t_new - t, torch.zeros_like(t))
            h = h_used.unsqueeze(1)

            k1 = f
            k2 = self._rhs(z + h * (k1 * (1.0 / 5.0)))
            k3 = self._rhs(z + h * (k1 * (3.0 / 40.0) + k2 * (9.0 / 40.0)))
            k4 = self._rhs(z + h * (k1 * (44.0 / 45.0) - k2 * (56.0 / 15.0) + k3 * (32.0 / 9.0)))
            k5 = self._rhs(z + h * (k1 * (19372.0 / 6561.0) - k2 * (25360.0 / 2187.0)
                                    + k3 * (64448.0 / 6561.0) - k4 * (212.0 / 729.0)))
            k6 = self._rhs(z + h * (k1 * (9017.0 / 3168.0) - k2 * (355.0 / 33.0)
                                    + k3 * (46732.0 / 5247.0) + k4 * (49.0 / 176.0)
                                    - k5 * (5103.0 / 18656.0)))
            z_new = z + h * (k1 * (35.0 / 384.0) + k3 * (500.0 / 1113.0) + k4 * (125.0 / 192.0)
                             - k5 * (2187.0 / 6784.0) + k6 * (11.0 / 84.0))
            k7 = self._rhs(z_new)
            err_vec = (k1 * (-71.0 / 57600.0) + k3 * (71.0 / 16695.0) - k4 * (71.0 / 1920.0)
                       + k5 * (17253.0 / 339200.0) - k6 * (22.0 / 525.0) + k7 * (1.0 / 40.0))
            sc = atol + torch.maximum(torch.abs(z), torch.abs(z_new)) * rtol
            err = self._rms(err_vec * h / sc)

            ok = err < 1.0
            accept = active & ok
            grow = torch.where(
                err == 0.0,
                torch.full_like(err, self.max_factor),
                torch.clamp(self.safety * torch.pow(err, ex), max=self.max_factor),
            )
            grow = torch.where(rejected, torch.clamp(grow, max=1.0), grow)
            shrink = torch.clamp(self.safety * torch.pow(err, ex), min=self.min_factor)
            h_abs = torch.where(active, h_used * torch.where(ok, grow, shrink), h_abs)

            a2 = accept.unsqueeze(1)
            t = torch.where(accept, t_new, t)
            z = torch.where(a2, z_new, z)
            f = torch.where(a2, k7, f)
            rejected = torch.where(accept, torch.zeros_like(rejected), rejected | (active & ~ok))
            done = done | (accept & (t_new >= T))
        return z

    # ------------------------------------------------------------------------
    def forward(
        self, Nc: Tensor, Nr: Tensor, qc: Tensor, qr: Tensor, muc: Tensor, mur: Tensor,
        dt: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns (dNr/dt, dNc/dt, dqc/dt, dqr/dt), each shaped like the broadcast inputs."""
        T = self.dt
        if dt is not None:
            T = dt
        if T <= 0.0:
            raise RuntimeError("dt must be positive")
        bt = torch.broadcast_tensors(Nc, Nr, qc, qr, muc, mur)
        out_shape = bt[0].shape
        out_dtype = bt[2].dtype
        wd = self.d_edges.dtype
        Nc_ = bt[0].reshape(-1).to(wd)
        Nr_ = bt[1].reshape(-1).to(wd)
        qc_ = bt[2].reshape(-1).to(wd)
        qr_ = bt[3].reshape(-1).to(wd)
        muc_ = bt[4].reshape(-1).to(wd)
        mur_ = bt[5].reshape(-1).to(wd)

        cloud_on = qc_ > self.qsmall
        rain_on = qr_ > self.qsmall

        dnc, dmc = self._bin_gamma(Nc_, qc_, muc_, cloud_on)
        dnr, dmr = self._bin_gamma(Nr_, qr_, mur_, rain_on)
        dndlnr = dnc + dnr
        dmdlnr = dmc + dmr
        M_dlnr = dmdlnr.sum(dim=-1)

        nr_t = torch.zeros_like(qc_)
        nc_t = torch.zeros_like(qc_)
        qc_t = torch.zeros_like(qc_)
        qr_t = torch.zeros_like(qc_)

        # Only run the ROM on columns that actually have liquid.
        idx = torch.nonzero((cloud_on | rain_on) & (M_dlnr > 0.0)).reshape(-1)
        if idx.numel() > 0:
            dn_b = dndlnr.index_select(0, idx)
            dm_b = dmdlnr.index_select(0, idx)
            M = M_dlnr.index_select(0, idx)

            x_scaled = dm_b / M.unsqueeze(1)
            z0 = torch.cat([self.encoder(x_scaled), (M / self.m_scale).unsqueeze(1)], dim=-1)
            if self.use_dopri5:
                z1 = self._integrate_dopri5(z0, T)
            else:
                z1 = self._integrate_rk4(z0, T)
            # decode; mass conserved exactly by construction (ignore latent M)
            dm_a = self.decoder(z1[:, :-1]) * M.unsqueeze(1)
            dn_a = dm_a * self.m_to_n.unsqueeze(0)

            cm = self.cloud_mask.unsqueeze(0)
            rm = 1.0 - cm
            c_dm = ((dm_a - dm_b) * cm).sum(dim=-1) * self.dlnr
            c_dn = ((dn_a - dn_b) * cm).sum(dim=-1) * self.dlnr
            r_dn = ((dn_a - dn_b) * rm).sum(dim=-1) * self.dlnr

            # Sign limiting (py_interface "LL Note"): only act when cloud mass
            # decreases; then qc, Nc may only decrease and qr = -qc.
            act = c_dm < 0.0
            zero = torch.zeros_like(c_dm)
            qct = torch.where(act, c_dm, zero) / T
            nct = torch.where(act, torch.clamp(c_dn, max=0.0), zero) / T
            nrt = torch.where(act, r_dn, zero) / T

            # Rain-only columns: there is no cloud category, so neither the
            # cloud/rain split nor the cloud-mass gate applies. qc, qr and Nc are
            # unchanged; Nr takes the total number change over all bins
            # (rain self-collection).
            rain_only = ((qc_ < self.qsmall) & rain_on).index_select(0, idx)
            nrt = torch.where(rain_only, (c_dn + r_dn) / T, nrt)
            nct = torch.where(rain_only, zero, nct)
            qct = torch.where(rain_only, zero, qct)

            # Nr may only grow where cloud droplets are lost (dNc/dt < 0);
            # otherwise dNr/dt <= 0. Covers rain-only columns, where dNc/dt = 0.
            if self.limit_nr:
                nrt = torch.where(nct < 0.0, nrt, torch.clamp(nrt, max=0.0))

            nr_t = nr_t.index_copy(0, idx, nrt)
            nc_t = nc_t.index_copy(0, idx, nct)
            qc_t = qc_t.index_copy(0, idx, qct)
            qr_t = qr_t.index_copy(0, idx, torch.zeros_like(qct) - qct)  # exact bulk mass conservation

        return (
            nr_t.reshape(out_shape).to(out_dtype),
            nc_t.reshape(out_shape).to(out_dtype),
            qc_t.reshape(out_shape).to(out_dtype),
            qr_t.reshape(out_shape).to(out_dtype),
        )