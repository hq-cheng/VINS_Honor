#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <map>
#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include "Parameters.h"

#include <assert.h>
#include <android/log.h>
#define LOG_TAG_FMANAGER "feature_manager.cc"
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO, LOG_TAG_FMANAGER, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG_FMANAGER, __VA_ARGS__)

// 示意图: 两个路标点p1, p2, 分别被图像1-2帧，2-4帧观测到
// landmarks         o  p1(x1, y1, z1)    o  p2(x2, y2, z2)
//                  .  .              .   .   .
//            _____.___  ._________.  ____.____  ._________
//           |    .    | | .    o  | |    .    | |  .      |
// frames    |   o     | |   o     | |    o    | |     .   |
//           |_________| |_________| |_________| |________o|
//             frame1      frame2      frame3      frame4

// VINS-Mono中的特征点管理器, 涉及3个类:
// FeaturePerId     管理一个特征点的属性
// FeaturePerFrame  管理一个特征点的帧属性
// FeatureManager   特征点管理器, 包含list<FeaturePerId>容器, 存储所有特征点的属性, 及其对应的帧属性

// FeaturePerFrame  管理一个特征点的帧属性
// 例如, 空间中路标点p1映射到frame1或frame2上对应的像素坐标、特征点速度、相机系归一化平面坐标等属性都封装到类 FeaturePerFrame 中
class FeaturePerFrame
{
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        cur_td = td;
    }
    double cur_td;          // IMU和相机不同步时的time offset
    Vector3d point;         // 特征点的相机系归一化平面坐标
    Vector2d uv;            // 特征点的像素坐标
    Vector2d velocity;      // 特征点速度
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
};

// FeaturePerId     管理一个特征点的属性
// 例如, 空间中路标点p1被两帧图像观测到，第一次观测到p1为frame1,即 start_frame=1，最后一次观测到p1为frame2,即 endframe()=2
// 数组vector<FeaturePerFrame>存储了 start_frame - endframe() 之间所有图像对应的帧属性
class FeaturePerId
{
  public:
    const int feature_id;                       // 特征点全局id
    int start_frame;                            // 观测到该特征点的第一帧图像序号
    vector<FeaturePerFrame> feature_per_frame;  // 该特征点帧属性数组, 观测到该特征点的所有图像对应的帧属性

    int used_num;           // 出现的次数, 等于feature_per_frame.size(), 即该特征点当前被 used_num 帧图像共同观测到
    bool is_outlier;
    bool is_margin;
    double estimated_depth; // 该特征点在"世界系"下深度估计值, 参考 triangulate() 函数
    int solve_flag;         // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Vector3d gt_p;

    // addFeatureCheckParallax() 函数中 start_frame 与第一次观测到该特征点图像的 frame_count 一致
    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    // 返回观测到该特征点的最后一帧图像序号
    int endFrame();
};

// FeatureManager 特征点管理器, 包含list<FeaturePerId>容器, 存储所有特征点的属性, 及其对应的帧属性
class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    // 特征点管理器中的外参ric初始化为单位矩阵, Estimator::setParameter()函数中调用 setRic() 函数修改
    void setRic(Matrix3d _ric[]);

    void clearState();

    // 获取 feature 数组中被两帧或两帧以上且都在滑动窗口队列内的图像同时观测到的特征点总个数
    int getFeatureCount();

    // 检查视差, 用于确定这一帧图像是否为关键帧
    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    void debugShow();
    // 给定两帧图像 frame_count = frame_count_l, frame_count = frame_count_r
    // 取出这两帧图像共同观测到的所有特征点的相机系归一化平面坐标 a, b, 并组合为 vector<pair<a, b>> 形式返回
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    // 给定feature 数组中被两帧或两帧以上且都在滑动窗口队列内的图像同时观测到的所有特征点的逆深度估计值向量 x
    // 设置对应所有特征点在世界系下的深度估计值, optimization()函数之后的double2vector()函数调用
    void setDepth(const VectorXd &x);
    // 移除掉 feature 数组中 solve_flag = 2的特征点, 即深度值为负数的特征点
    void removeFailures();
    void clearDepth(const VectorXd &x);
    // 获取 feature 数组中被两帧或两帧以上且都在滑动窗口队列内的图像同时观测到的所有特征点的逆深度估计值
    // 所有的逆深度估计值以向量 dep_vec 形式返回
    VectorXd getDepthVector();
    // 视觉特征三角化, 被visualInitialAlign()函数和solveOdometry()函数调用
    // 对滑动窗口队列中所有图像观测到的世界系深度 estimated_depth 未知的特征点进行三角化处理
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    list<FeaturePerId> feature; // 存储所有特征点的属性, 及其对应的帧属性
    int last_track_num;         // 说明这一帧图像有 last_track_num 个特征点在之前被其它的图像观测到过

  private:
    // 计算当前帧的前两帧图像(2nd last and 3rd last)之间的视差
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;
    Matrix3d ric[NUM_OF_CAM];   // 特征点管理器中的外参ric初始化为单位矩阵, Estimator::setParameter()函数中调用 setRic() 函数修改
};

#endif