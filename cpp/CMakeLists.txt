cmake_minimum_required(VERSION 3.18)
project(rom_demo CXX)
# Point CMake at LibTorch, e.g.
#   cmake -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)') ..
# or at an unpacked LibTorch download / your site install.
find_package(Torch REQUIRED)
add_executable(rom_demo main.cpp)
target_link_libraries(rom_demo "${TORCH_LIBRARIES}")
set_property(TARGET rom_demo PROPERTY CXX_STANDARD 17)
 