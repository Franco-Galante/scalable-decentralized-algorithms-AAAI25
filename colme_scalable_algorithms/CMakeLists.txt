if (UNIX)
    cmake_minimum_required(VERSION 3.14)
else()
    # Tested for Windows
    cmake_minimum_required(VERSION 3.26)
    cmake_policy(VERSION 3.26)
    cmake_policy(SET CMP0135 NEW)
endif()

project(DistributedMeanEstimation)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# Define __STDC_WANT_SECURE_LIB__
add_definitions(-D__STDC_WANT_SECURE_LIB__=1) 

add_executable(simulator main.cpp Helpers.cpp Rand.cpp)

if (UNIX)
    # Linux specific settings
    target_link_libraries(simulator PRIVATE stdc++fs)
endif()
