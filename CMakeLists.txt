# CMake 最低版本要求
cmake_minimum_required(VERSION 3.10) 

# 项目名称
project(opencv_demo)

# 查找OpenCV包
find_package(OpenCV REQUIRED)

# 设置C++标准为C++11
set(CMAKE_CXX_STANDARD 11)

set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# 自动遍历SOURCE_DIR目录下寻找所有cpp文件
file(GLOB SOURCE_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

# 添加可执行文件
add_executable(opencv_demo ${SOURCE_FILES}) 

# 将OpenCV库链接到目标可执行文件
target_link_libraries(opencv_demo ${OpenCV_LIBS}) 

# 加入OpenCV头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})