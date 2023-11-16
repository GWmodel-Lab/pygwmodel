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
        $<TARGET_FILE:py_gwr_basic>
        $<TARGET_FILE:py_spatial_weight>
        ${PYTHON_TEST_SCRIPT_DIR}/pygwmodel
    COMMAND ${CMAKE_COMMAND} -E
        copy_if_different
        ${PYGWM_PYTHON_TESTS}
        ${PYTHON_TEST_SCRIPT_DIR}
)

set(PYTHON_TEST_DATA ${CMAKE_CURRENT_SOURCE_DIR}/londonhp100.csv)
add_test(
    NAME testPythonGWRBasic
    COMMAND ${PYTHON_EXECUTABLE} testGWRBasic.py ${PYTHON_TEST_DATA}
)
# add_test(
#     NAME testPythonGWSS
#     COMMAND ${PYTHON_EXECUTABLE} testGWSS.py ${PYTHON_TEST_DATA}
# )
# add_test(
#     NAME testPythonGWPCA
#     COMMAND ${PYTHON_EXECUTABLE} testGWPCA.py ${PYTHON_TEST_DATA}
# )
