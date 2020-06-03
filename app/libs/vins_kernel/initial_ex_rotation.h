#pragma once 

#include <vector>
#include "Parameters.h"
#include "utility.h"
using namespace std;

#include <opencv2/opencv.hpp>

#include <eigen3/Eigen/Dense>

#include <android/log.h>
#define LOG_TAG_EXROTATION "initial_ex_rotation.cc"
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO, LOG_TAG_EXROTATION, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG_EXROTATION, __VA_ARGS__)

using namespace Eigen;

/* This class help you to calibrate extrinsic rotation between imu and camera when your totally don't konw the extrinsic parameter */
// 用于标定IMU与相机之间的外参, 主要是标定外参的旋转矩阵 ric
class InitialEXRotation
{
public:
    // 构造函数
	InitialEXRotation();
    // 在线进行IMU与相机之间的外参标定
    bool CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result);
private:
    // 利用对积几何约束求解前后两帧图像间的相对变换关系 R, t(以第一帧图像相机坐标系为参考坐标系)
	Matrix3d solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres);

    // 采用三角化对本质矩阵 E 分解得到的每组 R, t 进行验证，选择是特征点深度值为正的的 R, t
    double testTriangulation(const vector<cv::Point2f> &l,
                             const vector<cv::Point2f> &r,
                             cv::Mat_<double> R, cv::Mat_<double> t);
    
    // 对本质矩阵 E 进行SVD分解, 得到前后两帧图像间4组可能的变换关系 R, t
    void decomposeE(cv::Mat E,
                    cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                    cv::Mat_<double> &t1, cv::Mat_<double> &t2);
    int frame_count;            // 图像对序号( 两帧图像算一对 ), 初始化置 0

    vector< Matrix3d > Rc;      // 相邻两图像间的相机相对旋转矩阵 R, 由对极几何得到(以第一帧图像相机坐标系为参考坐标系)
    vector< Matrix3d > Rimu;    // 相邻两图像时刻之间的IMU相对旋转矩阵 R, 由IMU预积分得到(以第一帧图像时刻IMU坐标系为参考坐标系)
    vector< Matrix3d > Rc_g;    // 相邻两图像间的相机相对旋转矩阵 R, 由IMU预积分量 Rimu 与 外参 ric 求解得到
    Matrix3d ric;               // IMU与相机之间的外参旋转矩阵 ric
};


