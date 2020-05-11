cmake_minimum_required(VERSION 3.10.2)

add_definitions(-std=c++14)

set(CMAKE_C_COMPILER "gcc-7")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-7")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")


project(tracker)

find_package(OpenCV 4.1 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

find_package(PkgConfig)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp>=0.5)
include_directories(${YAMLCPP_INCLUDE_DIRS})


add_executable (tracker src/camFusion.cpp src/Tracker.cpp src/lidarData.cpp src/ReceiveData.cpp src/EgoVehicle.cpp src/ObjectDetection.h src/YoloV3.cpp src/FeaturePointMatcher2D.cpp)
target_link_libraries (tracker ${OpenCV_LIBRARIES} ${YAMLCPP_LIBRARIES})

