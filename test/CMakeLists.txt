cmake_minimum_required(VERSION 3.12)

if(ENABLE_OpenMP)
    set($ENV{ENABLE_OPENMP} "true")
endif(ENABLE_OpenMP)

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
