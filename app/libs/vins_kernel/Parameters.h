#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <string>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include "Camera.h"

using namespace camodocal;

// Global
enum DeviceType {
    Honor20,
    Unknown
};
#define NUM_OF_CAM 1
#define NUM_OF_F 1000
#define WINDOW_SIZE 10
#define STEREO_TRACK 0
extern int ROW;
extern int COL;
extern int FOCAL_LENGTH;
extern double INIT_DEPTH;
extern bool PUB_THIS_FRAME;

// Camera related parameters
extern Camera::ModelType CAMERA_TYPE;        // 相机类型
extern std::string CAMERA_NAME;              // 相机名称
extern cv::Size CAMERA_SIZE;                 // 图像尺寸
extern cv::Mat CAMERA_K;                     // 内参矩阵
extern cv::Mat CAMERA_D;                     // 畸变系数: k1, k2, p1, p2

// Extrinsic parameter between IMU and Camera
extern int ESTIMATE_EXTRINSIC;
extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;

// FeatureTracker related parameters
#define MAX_CNT 150
#define MIN_DIST 30
#define FREQ 10
#define F_THRESHOLD ((double)1.0)
#define SHOW_TRACK 1
#define EQUALIZE 1
#define FISHEYE 0

// Optimization related parameters
#define SOLVER_TIME ((double)0.04)
#define NUM_ITERATIONS 8
#define MIN_PARALLAX 10.0

// IMU related parameters
extern Eigen::Vector3d G;
extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

// Unsynchronization related parameters
#define ESTIMATE_TD 0
#define TD ((double)0.0)

// Rolling shutter related parameters
#define ROLLING_SHUTTER 0
#define TR ((double)0.0)

// Integration related parameters
enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};
enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};
enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};

bool SetParameters( DeviceType device);

#endif //PARAMETERS_H
