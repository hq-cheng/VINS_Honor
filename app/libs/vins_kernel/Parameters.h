#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <string>
#include <opencv2/core/core.hpp>
#include "Camera.h"

using namespace camodocal;

// FeatureTracker related parameters
#define MAX_CNT 150
#define MIN_DIST 30
#define FREQ 10
#define F_THRESHOLD ((double)1.0)
#define SHOW_TRACK 1
#define EQUALIZE 1
#define FISHEYE 0

#define NUM_OF_CAM 1
#define WINDOW_SIZE 10
#define STEREO_TRACK 0


extern int ROW;
extern int COL;
extern int FOCAL_LENGTH;
extern bool PUB_THIS_FRAME;

// Camera related parameters
extern Camera::ModelType CAMERA_TYPE;        // 相机类型
extern std::string CAMERA_NAME;              // 相机名称
extern cv::Size CAMERA_SIZE;                 // 图像尺寸
extern cv::Mat CAMERA_K;                     // 内参矩阵
extern cv::Mat CAMERA_D;                     // 畸变系数: k1, k2, p1, p2

enum DeviceType {
    Honor20,
    Unknown
};

bool SetParameters( DeviceType device);

#endif //PARAMETERS_H
