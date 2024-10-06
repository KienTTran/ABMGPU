cmake_minimum_required(VERSION 2.8.8)
project(easyloggingpp_builder C CXX)
include(ExternalProject)
ExternalProject_Add(easyloggingpp
    GIT_REPOSITORY https://github.com/muflihun/easyloggingpp.git
    CMAKE_ARGS -Dbuild_static_lib=ON 
    UPDATE_DISCONNECTED 1
    PREFIX ${CMAKE_BINARY_DIR}/ext
    #    PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
    # Disable install step
    INSTALL_COMMAND ""
    )

# Specify include dir
ExternalProject_Get_Property(easyloggingpp source_dir)
set(EASYLOGGINGPP_INCLUDE_DIR ${source_dir}/include PARENT_SCOPE)

# Specify MainTest's link libraries
ExternalProject_Get_Property(easyloggingpp binary_dir)
set(EASYLOGGINGPP_LIBS_DIR ${binary_dir} PARENT_SCOPE)