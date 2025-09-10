# cpp_SCREAM_python_ROM_interface
a demo wrapper that aims to communicate between C++ SCREAM and Python SDM-emulator, or reduced-order model (ROM). for now, it is tested on LLNL's Dane.

# NOTE on python:
Certain python packages need to be installed and activated within your local environment for this to work. To automatically create a local environment, run
```
./setup.sh
```
You made need to change the source / version of the torch module that is imported to work with your particular computing architecture.