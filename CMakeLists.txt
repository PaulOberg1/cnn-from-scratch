cmake_minimum_required(VERSION 3.10)
project(EggSpector)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Eigen path
set(EIGEN_PATH ${CMAKE_SOURCE_DIR}/eigen/eigen-3.4.0)

# Include directories
include_directories(${EIGEN_PATH} include)

# Source files
file(GLOB SOURCES "src/*.cpp")
file(GLOB SUBDIR_SOURCES "src/helper_functions/*.cpp")

set(SOURCES ${SOURCES} ${SUBDIR_SOURCES})


# Add executable target
add_executable(EggSpector ${SOURCES} src/main.cpp)

# Optionally, you can add targets for tests and examples
# Test targets
#add_executable(TestDenseLayer tests/TestDenseLayer.cpp)
#add_executable(TestConvLayer tests/TestConvLayer.cpp)

# Example targets
#add_executable(CNNExample examples/CNNExample.cpp)
