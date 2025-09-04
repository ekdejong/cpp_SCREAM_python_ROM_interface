import py_interface as rom

qc = 0.4e-3
nc = 1e9
qr = 0.2e-4
nr = 1e4
muc = 3
mur = 1
pressure_hPa = 850 * 100
temp_K = 280
rho = pressure_hPa/(temp_K*287.15)
mc = qc * rho
nc = nc * rho
mr = qr * rho
nr = nr * rho
qsmall = 1e-18

rom.ROM_interface(mc, nc, mr, nr, muc, mur, qsmall)

