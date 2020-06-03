#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "imu_factor.h"
#include "utility.h"
#include <map>
#include "feature_manager.h"

#include <assert.h>
#include <android/log.h>
#define LOG_TAG_INITALIGN "initial_alignment.cc"
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO, LOG_TAG_INITALIGN, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG_INITALIGN, __VA_ARGS__)

using namespace Eigen;
using namespace std;

class ImageFrame
{
    public:
        ImageFrame(){};
        ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, double _t):t{_t},is_key_frame{false}
        {
            points = _points;
        };
        map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>> > > points;
        double t;
        Matrix3d R;
        Vector3d T;
        IntegrationBase *pre_integration;
        bool is_key_frame;
};

// solveGyroscopeBias()函数利用旋转约束估计陀螺仪偏置, LinearAlignment()函数利用平移约束估计尺度, 重力加速度和速度
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x);