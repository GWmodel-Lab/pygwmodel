cmake_minimum_required(VERSION 3.12)
project(pygwmodel)

option(WITH_PYTESTS "Whether to build python tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir 
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(MSVC)

add_subdirectory(libgwmodel)

add_subdirectory(pygwmodel)

# if(WITH_PYTESTS)
#     enable_testing()
#     add_subdirectory(test)
# endif(WITH_PYTESTS)

