cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
# -O3 -Wall
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_BUILD_TYPE Release)

project(exam_test_v3)

add_definitions(-DUSE_MLU)
add_definitions(-DUSE_OPENCV)

include_directories("/usr/local/neuware/include")
link_directories("/usr/local/neuware/lib64")
set(CNRT_LIBS cnrt)

include_directories("/usr/include")
link_directories("/usr/lib/x86_64-linux-gnu")
set(LINKER_LIBS glog gflags)

include_directories("/home/tools/opencv-4.1.2/install/include/opencv4")
link_directories("/home/tools/opencv-4.1.2/install/lib")
set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc opencv_video opencv_videoio)

aux_source_directory(. DIR_SRCS)

file(GLOB SRCS  ./detect/*.cpp)
message(STASTUS "DIR: ${DIR_SRCS}  ${SRCS}")

add_executable(exam_test_v3 ${DIR_SRCS} ${SRCS})

target_link_libraries(exam_test_v3 ${LINKER_LIBS} ${CNRT_LIBS} ${OpenCV_LIBS} pthread)

set_property(TARGET exam_test_v3 PROPERTY CXX_STANDARD 11)
