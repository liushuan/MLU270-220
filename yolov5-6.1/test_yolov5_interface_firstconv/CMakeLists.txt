cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
# -O3 -Wall
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_BUILD_TYPE Release)

project(exam_test_v3)

add_definitions(-DUSE_MLU)
add_definitions(-DUSE_OPENCV)

include_directories("/root/tools/CambrianApp/3rd/neuware/include")
#link_directories("/root/tools/CambrianApp/3rd/neuware/lib64")
link_directories("/neuware/lib64")
set(CNRT_LIBS cnrt cndrv)

include_directories("/root/tools/CambrianApp/3rd/opencv/include")
link_directories("/root/tools/CambrianApp/3rd/opencv/lib")
set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc)

aux_source_directory(. DIR_SRCS)

file(GLOB SRCS  ./detect/*.cpp)
message(STASTUS "DIR: ${DIR_SRCS}  ${SRCS}")

add_executable(exam_test_v3 ${DIR_SRCS} ${SRCS})

target_link_libraries(exam_test_v3 ${CNRT_LIBS} ${OpenCV_LIBS} pthread)

set_property(TARGET exam_test_v3 PROPERTY CXX_STANDARD 11)
