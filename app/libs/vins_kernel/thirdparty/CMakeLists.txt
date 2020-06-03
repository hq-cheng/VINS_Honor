# Eigen
set(eigen_include_dir ${CMAKE_CURRENT_SOURCE_DIR}/eigen3)

set(thirdpartylibs_include_dir ${CMAKE_CURRENT_SOURCE_DIR})

add_library(eigen_lib INTERFACE)

target_include_directories(
    eigen_lib
    INTERFACE
    ${eigen_include_dir}
    ${thirdpartylibs_include_dir}
)

# Ceres Solver
set(ceres_include_dir ${CMAKE_CURRENT_SOURCE_DIR}/ceres-solver/include)
set(ceres_miniglog_include_dir ${CMAKE_CURRENT_SOURCE_DIR}/ceres-solver/internal/ceres/miniglog)
set(ceres_binary_dir ${CMAKE_CURRENT_SOURCE_DIR}/ceres-solver/obj/local/${ANDROID_ABI}/libceres.a)

add_library(
    ceres_lib
    STATIC
    IMPORTED
    GLOBAL
)

set_target_properties(
    ceres_lib
    PROPERTIES IMPORTED_LOCATION
    ${ceres_binary_dir}
)

set_target_properties(
    ceres_lib
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
    "${ceres_include_dir};${ceres_miniglog_include_dir}"
)

