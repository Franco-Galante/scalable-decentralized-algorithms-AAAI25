cmake_minimum_required(VERSION 3.26)
cmake_policy(VERSION 3.26)
cmake_policy(SET CMP0135 NEW)

project(DistributedMeanEstimation) # puoi usarla per il nome dell'eseguibile

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")


# argparse
include(FetchContent)
FetchContent_Declare(argparse URL https://github.com/p-ranav/argparse/archive/refs/tags/v2.9.tar.gz)
FetchContent_MakeAvailable(argparse)

include_directories(${argparse_SOURCE_DIR}/include)

# Define __STDC_WANT_SECURE_LIB__
add_definitions(-D__STDC_WANT_SECURE_LIB__=1) 

add_executable(simulator main.cpp Helpers.cpp Rand.cpp)

target_link_libraries(simulator PRIVATE argparse) # link, # ${PROJECT_NAME}