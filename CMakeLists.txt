
cmake_minimum_required(VERSION 3.10)
project(EggSpector)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Eigen path
set(EIGEN_PATH ${CMAKE_SOURCE_DIR}/eigen/eigen-3.4.0)

# Set OpenCV path to the OpenCV build directory
set(OpenCV_DIR "C:/opencv/opencv/build")  # Adjust this path to where OpenCVConfig.cmake is located

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${EIGEN_PATH} ${OpenCV_INCLUDE_DIRS} include)

# Source files
file(GLOB SOURCES "src/*.cpp")
file(GLOB SUBDIR_SOURCES "src/helper_functions/*.cpp")

set(SOURCES ${SOURCES} ${SUBDIR_SOURCES})

add_executable(EggSpector ${SOURCES} src/main.cpp)

# Link OpenCV libraries
target_link_libraries(EggSpector ${OpenCV_LIBS})