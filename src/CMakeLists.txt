cmake_minimum_required(VERSION 3.12)

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

set(LIBGWMODEL_INCLUDE_DIR ../libgwmodel/include)
include_directories(${LIBGWMODEL_INCLUDE_DIR}/gwmodelpp)

nanobind_add_module(_parallel STABLE_ABI parallel.cpp)
target_link_libraries(_parallel PRIVATE gwmodel)

nanobind_add_module(_spatial_weight STABLE_ABI spatial_weight.cpp)
target_link_libraries(_spatial_weight PRIVATE gwmodel)

nanobind_add_module(_regression STABLE_ABI common.hpp parallel.hpp base.cpp gwr_basic.cpp regression.cpp)
target_link_libraries(_regression PRIVATE gwmodel)

nanobind_add_module(_analysis STABLE_ABI common.hpp base.cpp parallel.hpp analysis.cpp gwss.cpp)
target_link_libraries(_analysis PRIVATE gwmodel)

install(TARGETS 
    _regression
    _spatial_weight
    _parallel
    _analysis
    LIBRARY DESTINATION pygwmodel)
