cmake_minimum_required (VERSION 3.5)

project(check_sa)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/SubProcesses/P1_Sigma_sm_gg_ttxgg/)
set(CONFIGURATION_NAME "Debug")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(HIPSYCL_PLATFORM "cpu")
set(HIPSYCL_INSTALL_LOCATION /opt/hipSYCL/)
set(hipSYCL_DIR ${HIPSYCL_INSTALL_LOCATION}/lib/cmake)

include_directories(
    src
    SubProcesses/P1_Sigma_sm_epem_mupmum
    ${HIPSYCL_INSTALL_LOCATION}/include
    ${HIPSYCL_INSTALL_LOCATION}/include/hipSYCL/contrib/
)

set (SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/read_slha.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/Parameters_sm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/rambo.cpp
    ${CMAKE_CURRENT_LIST_DIR}/SubProcesses/P1_Sigma_sm_gg_ttxgg/CPPProcess.cpp
    ${CMAKE_CURRENT_LIST_DIR}/SubProcesses/P1_Sigma_sm_gg_ttxgg/check_sa.cpp
)

add_definitions(-O0 -g -ferror-limit=5)

find_package(hipSYCL CONFIG REQUIRED)
add_executable(check_sa.exe ${SOURCES})
add_sycl_to_target(TARGET check_sa.exe SOURCES ${SOURCES})
