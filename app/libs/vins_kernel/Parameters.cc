#include "Parameters.h"

// FeatureTracker related parameters
int ROW;
int COL;
int FOCAL_LENGTH;
bool PUB_THIS_FRAME;

// Camera related parameters
Camera::ModelType CAMERA_TYPE;       // 相机类型
std::string CAMERA_NAME;             // 相机名称
cv::Size CAMERA_SIZE;                // 图像尺寸
cv::Mat CAMERA_K;                    // 内参矩阵
cv::Mat CAMERA_D;                    // 畸变系数: k1, k2, p1, p2

bool SetParameters( DeviceType device) {
    switch ( device ) {
        case Honor20:
            ROW = 480;
            COL = 640;
            FOCAL_LENGTH = 460;
            PUB_THIS_FRAME = false;

            CAMERA_TYPE = Camera::ModelType::PINHOLE,
            CAMERA_NAME = "camera",
            CAMERA_SIZE = cv::Size( 640, 480 );
            CAMERA_K = ( cv::Mat_<double>( 3, 3 ) << 478.7620, 0, 231.9446,
                                                      0, 478.6281, 319.5067,
                                                      0,        0,        1 );
            CAMERA_D = ( cv::Mat_<double>(4, 1) << 0.0, 0.0, 0.0, 0.0 );
            return true;
        case Unknown:
            return false;
        default:
            return false;
    }
}

