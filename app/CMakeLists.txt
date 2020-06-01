cmake_minimum_required(VERSION 3.4.1)

add_subdirectory(libs/vins_kernel)
add_subdirectory(src/main/cpp/vins_system)

add_library(
    native-lib
    SHARED
    src/main/cpp/ndk/native-lib.cc
)

find_library(
    log-lib
    log
)

target_link_libraries(
    native-lib
    vins_system
    log
)


