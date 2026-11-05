// Minimal C++ driver for the TorchScript AE-SINDy collision ROM.
// Build: see CMakeLists.txt.   Run: ./rom_demo ../../sdm_rom_r50um_dopri5.pt
//
// forward(Nc, Nr, qc, qr, mu_c, mu_r[, dt]) -> (dNr/dt, dNc/dt, dqc/dt, dqr/dt)
//   Nc, Nr [1/m3], qc, qr [kg/m3] (rho_air * mixing ratio), mu_c, mu_r [-]
//   dt [s] is optional; omitted -> the dt set at export.
#include <torch/script.h>

#include <cstdio>
#include <iostream>
#include <vector>

int main(int argc, const char* argv[]) {
  const char* path = argc > 1 ? argv[1] : "sdm_rom_r50um_dopri5.pt";

  torch::jit::script::Module rom;
  try {
    rom = torch::jit::load(path);  // add `rom.to(torch::kCUDA)` for GPU
  } catch (const c10::Error& e) {
    std::cerr << "could not load " << path << ":\n" << e.what() << "\n";
    return 1;
  }
  rom.eval();
  torch::NoGradGuard no_grad;

  // Values frozen into the model at export time
  std::printf("n_bins = %ld, r_threshold = %.2f um (split at r = %.2f um), default dt = %.0f s\n",
              (long)rom.attr("n_bins").toInt(), rom.attr("r_threshold").toDouble() * 1e6,
              rom.attr("r_split").toDouble() * 1e6, rom.attr("dt").toDouble());

  const double rho = 850.0 * 100.0 / (280.0 * 287.15);  // kg/m3

  // A batch of columns/levels: contiguous double arrays of length ncol.
  const int64_t ncol = 4;
  //                        col 0   col 1   col 2 (rain only)  col 3 (no liquid)
  std::vector<double> qc  = {1e-3,   1e-3,   0.0,               0.0};   // kg/kg
  std::vector<double> nc  = {1e8,    3e8,    1e8,               1e8};   // 1/kg
  std::vector<double> qr  = {1e-4,   1e-5,   1e-4,              0.0};   // kg/kg
  std::vector<double> nr  = {1e5,    1e4,    1e5,               1e5};   // 1/kg
  std::vector<double> muc = {3.0,    8.0,    3.0,               3.0};   // e.g. from P3's get_cloud_dsd2
  std::vector<double> mur = {1.0,    1.0,    1.0,               1.0};
  for (int64_t i = 0; i < ncol; ++i) { qc[i] *= rho; nc[i] *= rho; qr[i] *= rho; nr[i] *= rho; }

  auto opts = torch::TensorOptions().dtype(torch::kFloat64);
  // from_blob wraps existing memory without copying (e.g. a Kokkos view's data())
  auto t_Nc  = torch::from_blob(nc.data(),  {ncol}, opts);
  auto t_Nr  = torch::from_blob(nr.data(),  {ncol}, opts);
  auto t_qc  = torch::from_blob(qc.data(),  {ncol}, opts);
  auto t_qr  = torch::from_blob(qr.data(),  {ncol}, opts);
  auto t_muc = torch::from_blob(muc.data(), {ncol}, opts);
  auto t_mur = torch::from_blob(mur.data(), {ncol}, opts);

  for (double dt : {-1.0, 300.0}) {  // -1: use the exported default dt
    std::vector<torch::jit::IValue> in = {t_Nc, t_Nr, t_qc, t_qr, t_muc, t_mur};
    if (dt > 0.0) in.emplace_back(dt);
    auto out = rom.forward(in).toTuple();
    auto nr_tend = out->elements()[0].toTensor();  // 1/m3/s
    auto nc_tend = out->elements()[1].toTensor();  // 1/m3/s
    auto qc_tend = out->elements()[2].toTensor();  // kg/m3/s
    auto qr_tend = out->elements()[3].toTensor();  // kg/m3/s

    std::printf("%s\n", dt > 0.0 ? "dt = 300 s (passed at call time):" : "default dt:");
    for (int64_t i = 0; i < ncol; ++i) {
      // back to per-kg-air tendencies, as in the original driver
      std::printf("  col %ld: qctend = %-+.6E  qrtend = %-+.6E  nctend = %-+.6E  nrtend = %-+.6E\n",
                  (long)i, qc_tend[i].item<double>() / rho, qr_tend[i].item<double>() / rho,
                  nc_tend[i].item<double>() / rho, nr_tend[i].item<double>() / rho);
    }
  }
  return 0;
}