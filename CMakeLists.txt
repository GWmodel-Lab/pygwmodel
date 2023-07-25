cmake_minimum_required(VERSION 3.12)
project(pygwmodel)

option(WITH_PYTESTS "Whether to build python tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})


if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(MSVC)


add_subdirectory(libgwmodel)

add_subdirectory(pygwmodel)

if(WITH_PYTESTS)
    enable_testing()
    add_subdirectory(test)
endif(WITH_PYTESTS)

