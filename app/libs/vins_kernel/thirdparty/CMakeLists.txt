set(eigen_include_dir ${CMAKE_CURRENT_SOURCE_DIR}/eigen3)

set(thirdpartylibs_include_dir ${CMAKE_CURRENT_SOURCE_DIR})

add_library(eigen_lib INTERFACE)

target_include_directories(
    eigen_lib
    INTERFACE
    ${eigen_include_dir}
    ${thirdpartylibs_include_dir}
)