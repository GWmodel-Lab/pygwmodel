cmake_minimum_required(VERSION 3.12)
project(pygwmodel)

option(WITH_PYTESTS "Whether to build python tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Armadillo REQUIRED)
if(ARMADILLO_FOUND)
    add_definitions(-DARMA_USE_LAPACK -DARMA_USE_BLAS -DARMA_DONT_USE_WRAPPER)
    include_directories(${ARMADILLO_INCLUDE_DIRS})

    find_package(BLAS)
    if(NOT BLAS_FOUND)
        message(SEND_ERROR "BLAS Not Found!")
    endif(NOT BLAS_FOUND)
    
    find_package(LAPACK)
    if(NOT LAPACK_FOUND)
        message(SEND_ERROR "LAPACK Not Found!")
    endif(NOT LAPACK_FOUND)
else(ARMADILLO_FOUND)
    message(FATAL_ERROR "Armadillo Not Found!")
endif(ARMADILLO_FOUND)
if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(MSVC)
set(LIBGWMODEL_INCLUDE_DIR libgwmodel/include)
include_directories(${LIBGWMODEL_INCLUDE_DIR}/gwmodelpp)
add_subdirectory("libgwmodel")
add_subdirectory("pygwmodel")

if(WITH_PYTESTS)
    enable_testing()
    add_subdirectory("test")
endif(WITH_PYTESTS)

set(PYGWMODEL_SKBUILD_DIR ${CMAKE_BINARY_DIR}/pygwmodel)
add_custom_target(pygwmodel_skbuild
    VERBATIM
    COMMAND_EXPAND_LISTS
    COMMAND ${CMAKE_COMMAND} -E
        make_directory
        ${PYGWMODEL_SKBUILD_DIR}
    COMMAND ${CMAKE_COMMAND} -E
        copy_directory ${CMAKE_SOURCE_DIR}/python ${PYGWMODEL_SKBUILD_DIR}
    COMMAND ${CMAKE_COMMAND} -E
        copy_directory ${CMAKE_SOURCE_DIR}/cmake ${PYGWMODEL_SKBUILD_DIR}/cmake
    COMMAND ${CMAKE_COMMAND} -E
        copy_directory ${CMAKE_SOURCE_DIR}/libgwmodel/include ${PYGWMODEL_SKBUILD_DIR}/include
    COMMAND ${CMAKE_COMMAND} -E
        copy_directory ${CMAKE_SOURCE_DIR}/libgwmodel/src ${PYGWMODEL_SKBUILD_DIR}/src
)

include_directories(${LIBGWMODEL_INCLUDE_DIR}/gwmodelpp)
