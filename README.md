# cpp_SCREAM_python_ROM_interface
a demo wrapper that aims to communicate between C++ SCREAM and Python SDM-emulator, or reduced-order model (ROM). for now, it is tested on LLNL's Dane.

# TorchScript port of the SCREAM ↔ AE-SINDy collision ROM interface

| file | purpose |
|---|---|
| `rom_torch.py` | `SDMCollisionROM`: pure-PyTorch, scriptable replacement for `py_interface.py` + `py_ROM.py` + `rom_utils.py` |
| `export_rom.py` | freezes the config (radius threshold, dt, μc, μr, qsmall, integrator), scripts, checks and saves the `.pt` |
| `cpp/` | LibTorch driver (`main.cpp`) + `CMakeLists.txt` |

```bash
python export_rom.py --r-threshold-um 50          # -> sdm_rom_r40um_dopri5.pt
cd cpp
cmake -B build -DTorch_DIR=/usr/workspace/wsa/dejong5/software/libtorch/share/cmake/Torch
cmake --build build
./build/rom_demo ../sdm_rom_r50um_dopri5.pt
```

n_bins, latent size and SINDy polynomial order are read from the weights file. The defaults (aesindy_weights.pth, 64 bins, zlim_data.npz, M_SCALE) are the erfAll model; --weights aesindy_weights_pysdm.pth gives the 63-bin box model. The bin grid (--r-min, --r-max), --m-scale and --zlim are not stored in the weights and must match what each model was trained with.

Interface of the saved module: forward(Nc, Nr, qc, qr, mu_c, mu_r, dt=None) -> (dNr/dt, dNc/dt, dqc/dt, dqr/dt). Nc, Nr, qc, qr are per volume of air (1/m³, kg/m³); mu_c and mu_r are per-cell tensors (scalars broadcast); dt is an optional float that defaults to the exported value. All tensors may have any shape and are float64 by default; outputs have the broadcast shape, in 1/m³/s and kg/m³/s. Tendencies follow py_interface.py's sign limiting: everything is zero unless cloud-bin mass decreases, and then dqc/dt ≤ 0, dNc/dt ≤ 0, dqr/dt = −dqc/dt. Rain-only cells (qc < qsmall < qr) are the exception: dqc/dt = dqr/dt = dNc/dt = 0 and dNr/dt is the total number change over all bins, not gated by the cloud-mass test. In every cell, dNr/dt can be positive only where dNc/dt < 0; otherwise it is capped at 0 (so rain-only cells get dNr/dt ≤ 0). --no-limit-nr turns this off. Frozen settings are module attributes (n_bins, r_threshold, r_split, cutoff_idx, dt, qsmall).

# NOTE on python:
Certain python packages need to be installed and activated within your local environment for this to work. To automatically create a local environment, run
```
./setup.sh
```
You made need to change the source / version of the torch module that is imported to work with your particular computing architecture.