cmake_minimum_required(VERSION 2.8.9)
project (test)

set (OPENCV4_ROOT "c:/msys64/usr/local" )

message("echo $ENV{OPENCV4_INCLUDE}")

SET (MNN_ROOT ./ )

set (CMAKE_BUILD_TYPE release)
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS}  -O3 -Wall -DNDEBUG")

include_directories( ${MNN_ROOT}/include ./3rd_party/imageHelper include ${OPENCV4_ROOT}/include )
file (GLOB TEST_SRC ./*.c ./*.cpp)

set ( PROJECT_LINK_LIBS libMNN.a libopencv_world411.dll.a )
link_directories( ${MNN_ROOT}/lib ${OPENCV4_ROOT}/lib )

add_executable(test ${TEST_SRC})
target_link_libraries(test ${PROJECT_LINK_LIBS} )

