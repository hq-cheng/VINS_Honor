cmake_minimum_required(VERSION 3.4.1)
project(VINS_Honor)

# done here so it's not necessary for every target_link_library call of opencv
set(OpenCV_DIR D:/Android/opencv430/OpenCV-android-sdk/sdk/native/jni)
find_package(OpenCV REQUIRED)

add_subdirectory(thirdparty)

set(
    VINS_SRCS
    draw_result.cc
    utility.cc
    Parameters.cc
    Camera.cc
    CameraFactory.cc
    PinholeCamera.cc
    feature_tracker.cc
    estimator.cc
    feature_manager.cc
    solve_5pts.cc
    initial_sfm.cc
    initial_aligment.cc
    initial_ex_rotation.cc
    pose_local_parameterization.cc
    projection_factor.cc
    projection_td_factor.cc
    marginalization_factor.cc
)

set(TARGET_NAME vins_kernel)

add_library(${TARGET_NAME} STATIC ${VINS_SRCS})

target_include_directories(${TARGET_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
    ${TARGET_NAME}
    eigen_lib
    ceres_lib
    log
    ${OpenCV_LIBS}
)