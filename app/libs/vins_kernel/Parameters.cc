#include "Parameters.h"

// Global
int ROW;
int COL;
int FOCAL_LENGTH;
double INIT_DEPTH;
bool PUB_THIS_FRAME;

// Camera related parameters
Camera::ModelType CAMERA_TYPE;       // 相机类型
std::string CAMERA_NAME;             // 相机名称
cv::Size CAMERA_SIZE;                // 图像尺寸
cv::Mat CAMERA_K;                    // 内参矩阵
cv::Mat CAMERA_D;                    // 畸变系数: k1, k2, p1, p2

// Extrinsic parameter between IMU and Camera
int ESTIMATE_EXTRINSIC;
std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

// IMU related parameters
Eigen::Vector3d G;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

bool SetParameters( DeviceType device) {
    switch ( device ) {
        case Honor20:
            ROW = 480;
            COL = 640;
            FOCAL_LENGTH = 460;
            INIT_DEPTH = 5.0;
            PUB_THIS_FRAME = false;

            CAMERA_TYPE = Camera::ModelType::PINHOLE,
            CAMERA_NAME = "camera",
            CAMERA_SIZE = cv::Size( 640, 480 );
            CAMERA_K = ( cv::Mat_<double>( 3, 3 ) << 478.7620, 0, 231.9446,
                                                      0, 478.6281, 319.5067,
                                                      0,        0,        1 );
            CAMERA_D = ( cv::Mat_<double>(4, 1) << 0.0, 0.0, 0.0, 0.0 );

            if ( ESTIMATE_EXTRINSIC == 2 ) {
                RIC.push_back(Eigen::Matrix3d::Identity());
                TIC.push_back(Eigen::Vector3d::Zero());
            }
            else {
                cv::Mat cv_R, cv_T;
                cv_R = ( cv::Mat_<double>(3,3) << 0.015, -0.999, 0.004,
                                                  0.999,  0.015, 0.026,
                                                 -0.026,  0.004, 0.999);
                cv_T = ( cv::Mat_<double>(3,1) << 0.0,
                                              -0.0045,
                                             -0.01505);
                Eigen::Matrix3d eigen_R;
                Eigen::Vector3d eigen_T;
                cv::cv2eigen(cv_R, eigen_R);
                cv::cv2eigen(cv_T, eigen_T);
                Eigen::Quaterniond Q(eigen_R);
                eigen_R = Q.normalized().toRotationMatrix();
                RIC.push_back(eigen_R);
                TIC.push_back(eigen_T);
            }

            G << 0, 0, 9.81007;
            ACC_N = 0.2;
            ACC_W = 0.002;
            GYR_N = 0.2;
            GYR_W = 4.0e-5;

            return true;
        case Unknown:
            return false;
        default:
            return false;
    }
}

