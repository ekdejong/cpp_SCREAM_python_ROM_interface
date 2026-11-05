#!/usr/bin/env python3
"""
Build the AE-SINDy collision ROM with a fixed configuration, compile it to
TorchScript, check the compiled module against eager mode, and save it.

    python export_rom.py --r-threshold-um 50 --out sdm_rom_r50um.pt

The radius threshold, qsmall, default dt and integrator settings are baked into
the saved file. The C++ side passes (Nc, Nr, qc, qr, mu_c, mu_r[, dt]).
"""
import argparse
import warnings

import torch

from rom_torch import SDMCollisionROM


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="ftorch_weights/erfAll_aesindy_weights.pth", help="erfAll (64 bins) by default")
    p.add_argument("--zlim", default="ftorch_weights/zlim_data.npz")
    p.add_argument("--r-threshold-um", type=float, default=50.0, help="cloud/rain split radius [um]")
    p.add_argument("--dt", type=float, default=100.0, help="default ROM integration window [s]; can be overridden per call")
    p.add_argument("--qsmall", type=float, default=1e-18)
    p.add_argument("--integrator", choices=["dopri5", "rk4"], default="dopri5")
    p.add_argument("--rtol", type=float, default=1e-3, help="dopri5 rtol (1e-3 = scipy default)")
    p.add_argument("--atol", type=float, default=1e-6, help="dopri5 atol (1e-6 = scipy default)")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--n-substeps", type=int, default=10, help="rk4 substeps")
    p.add_argument("--r-min", type=float, default=1e-6, help="smallest bin edge radius [m]")
    p.add_argument("--r-max", type=float, default=5e-3, help="largest bin edge radius [m]")
    p.add_argument("--m-scale", type=float, default=0.01046717958952477,
                   help="mass normalization used in training [kg m^-3 (ln r)^-1]")
    p.add_argument("--no-limit-nr", action="store_true",
                   help="disable the rule that dNr/dt > 0 only where dNc/dt < 0")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    model = SDMCollisionROM(
        a.weights, a.zlim,
        r_threshold=a.r_threshold_um * 1e-6, dt=a.dt, qsmall=a.qsmall,
        integrator=a.integrator, rtol=a.rtol, atol=a.atol, max_steps=a.max_steps,
        n_substeps=a.n_substeps, limit_nr=not a.no_limit_nr, r_min=a.r_min, r_max=a.r_max, m_scale=a.m_scale,
        dtype=getattr(torch, a.dtype),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # torch.jit.script deprecation notice
        scripted = torch.jit.script(model)

    # sanity check: scripted == eager on the main.cpp test state (per-volume units)
    rho = 850e2 / (280.0 * 287.15)
    x = [torch.tensor([v * rho], dtype=torch.float64) for v in (1e7, 1e4, 1e-4, 1e-4)]
    x += [torch.tensor([3.0], dtype=torch.float64), torch.tensor([1.0], dtype=torch.float64)]  # mu_c, mu_r
    with torch.no_grad():
        ref = torch.stack(model(*x))
        out = torch.stack(scripted(*x))
    assert torch.allclose(ref, out, rtol=1e-12, atol=0.0), (ref, out)

    out_path = a.out or f"sdm_rom_r{a.r_threshold_um:g}um_{a.integrator}.pt"
    scripted.save(out_path)

    # reload and check the file round-trips
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        loaded = torch.jit.load(out_path)
    with torch.no_grad():
        assert torch.equal(torch.stack(loaded(*x)), out)

    print(f"saved {out_path}")
    print(f"  weights {a.weights}: n_bins = {model.n_bins}, n_latent = {model.n_latent}, "
          f"poly_order = {model.poly_order}")
    print(f"  r_threshold = {model.r_threshold * 1e6:g} um -> cloud = bins 0..{model.cutoff_idx}, "
          f"split at r = {model.r_split * 1e6:.2f} um")
    print(f"  default dt = {model.dt:g} s, qsmall = {model.qsmall:g}, "
          f"integrator = {a.integrator}, dtype = {a.dtype}, "
          f"limit_nr = {model.limit_nr}")
    names = ["dNr/dt [1/m3/s]", "dNc/dt [1/m3/s]", "dqc/dt [kg/m3/s]", "dqr/dt [kg/m3/s]"]
    for n, v in zip(names, out.reshape(-1).tolist()):
        print(f"  test state (mu_c=3, mu_r=1)  {n:18s} {v:+.10e}")


if __name__ == "__main__":
    main()