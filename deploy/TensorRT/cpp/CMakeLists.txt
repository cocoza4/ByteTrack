cmake_minimum_required(VERSION 3.13)

project(bytetrack)

add_definitions(-std=c++14)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/include/eigen3)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(PkgConfig) 
pkg_search_module(GLIB REQUIRED glib-2.0) 
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
# pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)
pkg_check_modules(GST_VIDEO REQUIRED gstreamer-video-1.0)
# target_include_directories(gstreamer PRIVATE ${GTK3_INCLUDE_DIRS} ${GST_INCLUDE_DIRS})
message(STATUS "xxxxxxxxxxx ${GSTREAMER_INCLUDE_DIRS} yyyy ${GST_VIDEO_LIBRARIES}")
include_directories(${GSTREAMER_INCLUDE_DIRS})

# deepstream lib path
link_directories(/opt/nvidia/deepstream/deepstream/lib)
include_directories(/opt/nvidia/deepstream/deepstream/sources/includes)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# file(GLOB My_Source_Files ${PROJECT_SOURCE_DIR}/src/*.cpp)
set(COMMON_FILES src/BYTETracker.cpp;src/kalmanFilter.cpp;src/lapjv.cpp;src/STrack.cpp;src/utils.cpp)
add_library(common STATIC ${COMMON_FILES})
set(COMMON_LIBS common;cudart;${OpenCV_LIBS};${GST_APP_LIBRARIES};${GST_VIDEO_LIBRARIES})

add_executable(bytetrack src/bytetrack.cpp)
target_link_libraries(bytetrack PRIVATE ${COMMON_LIBS} nvinfer)

add_executable(ds_bytetrack src/ds_bytetrack.cpp)
target_link_libraries(ds_bytetrack PRIVATE ${COMMON_LIBS} nvds_meta nvdsgst_meta)

add_executable(ds_rtsp_bytetrack src/ds_rtsp_bytetrack.cpp)
target_link_libraries(ds_rtsp_bytetrack ${COMMON_LIBS} nvds_meta nvdsgst_meta)

add_definitions(-O2 -pthread)

