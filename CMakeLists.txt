cmake_minimum_required(VERSION 3.16)
project(ExamMonitorSystem VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Mặc định Release (-O3): bản đóng gói trước đây build Debug (-O0, không tối ưu) do không chỉ định
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ============================================================
# Qt6 
# ============================================================
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

if(DEFINED ENV{LOCAL_QT_DIR})
    set(CMAKE_PREFIX_PATH "$ENV{LOCAL_QT_DIR}")
else()
    set(CMAKE_PREFIX_PATH "C:/Qt/6.11.1/mingw_64")
endif()

find_package(Qt6 REQUIRED COMPONENTS Widgets)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Fix UTF-8 Encoding cho MinGW để hiển thị đúng các Icon (Emoji) và Tiếng Việt
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options("-fexec-charset=utf-8")
    add_compile_options("-finput-charset=utf-8")
endif()

# ============================================================
# MSYS2 Paths (Use mingw64 for MSVCRT ABI compatibility with Qt)
# ============================================================
if(DEFINED ENV{LOCAL_MSYS2_DIR})
    set(MSYS2_ROOT "$ENV{LOCAL_MSYS2_DIR}")
else()
    set(MSYS2_ROOT "C:/msys64/mingw64")
endif()
list(APPEND CMAKE_PREFIX_PATH ${MSYS2_ROOT})

# ============================================================
# Protobuf (from MSYS2 mingw64 via pkg-config)
# ============================================================
find_package(PkgConfig REQUIRED)
pkg_check_modules(PROTOBUF REQUIRED protobuf)
include_directories(${PROTOBUF_INCLUDE_DIRS})
add_compile_definitions(PROTOBUF_USE_DLLS)

# ============================================================
# OpenCV (Local Build)
# ============================================================
if(DEFINED ENV{LOCAL_OPENCV_DIR})
    set(OpenCV_DIR "$ENV{LOCAL_OPENCV_DIR}")
else()
    set(OpenCV_DIR "D:/CodeSpace/Work_Space/thesis_iuh/opencv_build")
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ============================================================
# ZeroMQ (from MSYS2 mingw64)
# ============================================================
find_package(cppzmq REQUIRED)

# ============================================================
# Source Files
# ============================================================
include_directories(proto/cpp_out)
include_directories(cpp_client/src cpp_client/ui)

set(PROTO_SRC
    proto/cpp_out/exam_monitor.pb.cc
)

set(CORE_SRC
    cpp_client/src/main.cpp
    cpp_client/src/system_manager.hpp
    cpp_client/src/system_manager.cpp
    cpp_client/src/video_capture.hpp
    cpp_client/src/video_capture.cpp
    cpp_client/src/zmq_publisher.hpp
    cpp_client/src/zmq_publisher.cpp
    cpp_client/src/zmq_subscriber.hpp
    cpp_client/src/zmq_subscriber.cpp
    cpp_client/src/image_processor.hpp
    cpp_client/src/image_processor.cpp
    cpp_client/src/emotion_worker.hpp
    cpp_client/src/emotion_worker.cpp
    cpp_client/src/behavior_labels.hpp
    cpp_client/src/app_settings.hpp
    cpp_client/src/report_store.hpp
    cpp_client/src/report_store.cpp
)

set(UI_SRC
    cpp_client/ui/main_window.hpp
    cpp_client/ui/main_window.cpp
    cpp_client/ui/camera_widget.hpp
    cpp_client/ui/camera_widget.cpp
    cpp_client/ui/ui_helpers.hpp
    cpp_client/ui/live_page.hpp
    cpp_client/ui/live_page.cpp
    cpp_client/ui/history_page.hpp
    cpp_client/ui/history_page.cpp
    cpp_client/ui/stats_page.hpp
    cpp_client/ui/stats_page.cpp
    cpp_client/ui/settings_page.hpp
    cpp_client/ui/settings_page.cpp
    cpp_client/resources/resources.qrc
)

add_executable(${PROJECT_NAME} 
    ${PROTO_SRC}
    ${CORE_SRC}
    ${UI_SRC}
)

# ============================================================
# Link Libraries
# ============================================================

# KHÔNG thêm -L C:/msys64/mingw64/lib: thư mục này chứa CRT (libmingwex, libmsvcrt...) của MSYS2, khác bản đi kèm
# trình biên dịch Qt mingw1310 -> atexit/_onexit gọi vòng lẫn nhau, app tràn stack (0xc00000fd) ngay khi mở.
# Protobuf link bằng đường dẫn tuyệt đối (PROTOBUF_LINK_LIBRARIES) thay cho -L + tên thư viện.
target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Widgets
    ${OpenCV_LIBS}
    cppzmq
    ${PROTOBUF_LINK_LIBRARIES}
)