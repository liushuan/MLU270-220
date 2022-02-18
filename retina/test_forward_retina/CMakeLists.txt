cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
# -O3 -Wall
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_BUILD_TYPE Release)


project(retina_test)

add_definitions(-DUSE_MLU)
add_definitions(-DUSE_OPENCV)

include_directories("/usr/local/neuware/include")
link_directories("/usr/local/neuware/lib64")
set(CNRT_LIBS cnrt)

#include_directories("/usr/include")
#link_directories("/usr/lib/x86_64-linux-gnu")
#set(LINKER_LIBS glog gflags)

include_directories("/usr/include/eigen3")
add_definitions(-DUSE_NEON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")


include_directories("/home/tools/opencv-4.1.2/install/include/opencv4")
link_directories("/home/tools/opencv-4.1.2/install/lib")
set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc opencv_video opencv_videoio)

aux_source_directory(. DIR_SRCS)

file(GLOB SRCS ./track/DeepAppearanceDescriptor/*.cpp ./track/KalmanFilter/*.cpp ./track/MunkresAssignment/*.cpp ./track/MunkresAssignment/munkres/*.cpp ./detect/*.cpp ./task/*.cpp ./decoder/*.cpp)
message(STASTUS "DIR: ${DIR_SRCS}  ${SRCS}")

add_executable(retina_test ${DIR_SRCS} ${SRCS})

target_link_libraries(retina_test ${LINKER_LIBS} ${CNRT_LIBS} ${OpenCV_LIBS} pthread)

set_property(TARGET retina_test PROPERTY CXX_STANDARD 11)
