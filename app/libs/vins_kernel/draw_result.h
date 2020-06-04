#ifndef DRAW_RESULT_H
#define DRAW_RESULT_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include "utility.h"
#include "Parameters.h"

using namespace cv;
using namespace Eigen;
using namespace std;

#define HEIGHT 480
#define WIDTH 640

class DrawResult {
public:
    DrawResult( float _yall, float _pitch, float _roll, float _T_x, float _T_y, float _T_z );
    ~DrawResult();

    float theta, phy;
    float radius;
    Vector3f origin_w;
    float Fx,Fy;
    float X0, Y0;

    float yaw, pitch, roll;
    float Tx, Ty, Tz; //in meters, only in z axis
    std::vector<Vector3f> pose;

    bool checkBorder(const cv::Point2f &pt);
    std::vector<Eigen::Vector3f> CalculateCameraPose( Eigen::Vector3f camera_center, Eigen::Matrix3f Rc, float length );
    cv::Point2f World2VirturCam( Eigen::Vector3f xyz, float &depth );
    void Reprojection( cv::Mat& result, std::vector<Eigen::Vector3f>& point_cloud,
                       const Eigen::Matrix3f* R_window, const Eigen::Vector3f *t_window );
};


#endif //DRAW_RESULT_H
