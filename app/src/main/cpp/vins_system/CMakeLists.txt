set(OpenCV_DIR D:/Android/opencv430/OpenCV-android-sdk/sdk/native/jni)
find_package(OpenCV REQUIRED)

add_library(
    vins_system
    STATIC
    system.cc
)

target_link_libraries(
    vins_system
    vins_kernel
    android
    log
    ${OpenCV_LIBS}
)

target_include_directories(vins_system INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})