cmake_minimum_required(VERSION 3.8)
project(gmrt_vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TowerStatus.msg"
  DEPENDENCIES std_msgs
)

# vision_node.py di-install sebagai executable, dijalankan lewat:
#   ros2 run gmrt_vision vision_node.py --ros-args -p source:=0
install(PROGRAMS
  scripts/vision_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  sample_videos
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
