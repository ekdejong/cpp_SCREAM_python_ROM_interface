#include <pybind11/embed.h>
#include <iostream>

namespace py = pybind11;

int main() {
    // Start Python interpreter
    py::scoped_interpreter guard{};  

    // Pass dummy cloud variables to SDM interface
    double qc = 0.4e-3;    // unit: kg/kg
    double nc = 1e9;       // unit: 1/kg
    double qr = 0.2e-4;    // unit: kg/kg
    double nr = 1e4;       // unit: 1/kg
    double muc = 3;        // unitless
    double mur = 1;        // unitless

    // convert unit
    double pressure_hPa = 850*100.;  // unit: Pa
    double temp_K =  280;            // unit: Kelvin

    double rho = pressure_hPa/(temp_K*287.15);  // unit: kg/m3

    double qc2 = qc * rho;  // unit: kg/m3
    double nc2 = nc * rho;  // unit: 1/m3
    double qr2 = qr * rho;  // unit: kg/m3
    double nr2 = nr * rho;  // unit: 1/m3

    // update to use SCREAM default qsmall when implementation.
    double QSMALL = 1E-18;

    // Add current dir to Python path
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(1, ".");

    std::cout << "C++ Begins. " << std::endl;
    // Import SDM_interface python module and call function with scalars
    py::module mod = py::module::import("py_interface");
    py::tuple result = mod.attr("SDM_interface")(qc2, nc2, qr2, nr2, muc, mur, QSMALL);

    // Unpack tuple: retrieve tendency rate terms
    double qctend = result[0].cast<double>()/rho;  // unit: kg/kg/s
    double nctend = result[1].cast<double>()/rho;  // unit: 1/kg/s
    double qrtend = result[2].cast<double>()/rho;  // unit: kg/kg/s
    double nrtend = result[3].cast<double>()/rho;  // unit: 1/kg/s
  
    
    std::cout << "Python returned: " << qctend << ", " << nctend << ", " << qrtend << ", " << nrtend << std::endl;

    printf("After call: qctend = %-+.16E, qrtend = %-+.16E, nctend = %-+.16E, nrtend = %-+.16E\n", qctend, qrtend, nctend, nrtend);
}