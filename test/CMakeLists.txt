cmake_minimum_required(VERSION 3.12)

if(ENABLE_OpenMP)
    set($ENV{ENABLE_OPENMP} "true")
endif(ENABLE_OpenMP)

if(WIN32)
set(PYTHON_TEST_SCRIPT_DIR ${pygwmodel_BINARY_DIR}/bin/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>)
else()
set(PYTHON_TEST_SCRIPT_DIR ${pygwmodel_BINARY_DIR}/bin)
endif()

file(GLOB PYGWM_PYTHON_SCRIPTS "${CMAKE_CURRENT_SOURCE_DIR}/../src/pygwmodel/*.py")
file(GLOB PYGWM_PYTHON_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/test*.py")

add_custom_target(testPythonDeps ALL 
    VERBATIM 
    COMMAND_EXPAND_LISTS
    COMMAND ${CMAKE_COMMAND} -E
        make_directory
        ${PYTHON_TEST_SCRIPT_DIR}/pygwmodel
    COMMAND ${CMAKE_COMMAND} -E
        copy_if_different
        ${PYGWM_PYTHON_SCRIPTS}
        $<TARGET_FILE:_spatial_weight>
        $<TARGET_FILE:_parallel>
        $<TARGET_FILE:_regression>
        $<TARGET_FILE:_analysis>
        ${PYTHON_TEST_SCRIPT_DIR}/pygwmodel
    COMMAND ${CMAKE_COMMAND} -E
        copy_if_different
        ${PYGWM_PYTHON_TESTS}
        ${PYTHON_TEST_SCRIPT_DIR}
)

set(PYTHON_TEST_DATA ${CMAKE_CURRENT_SOURCE_DIR}/londonhp100.csv)
add_test(
    NAME testPythonGWRBasic
    COMMAND ${Python_EXECUTABLE} test_gwr_basic.py ${PYTHON_TEST_DATA}
    WORKING_DIRECTORY ${PYTHON_TEST_SCRIPT_DIR}
)
add_test(
    NAME testPythonGWSS
    COMMAND ${Python_EXECUTABLE} test_gwss.py ${PYTHON_TEST_DATA}
    WORKING_DIRECTORY ${PYTHON_TEST_SCRIPT_DIR}
)
# add_test(
#     NAME testPythonGWPCA
#     COMMAND ${Python_EXECUTABLE} testGWPCA.py ${PYTHON_TEST_DATA}
# )
