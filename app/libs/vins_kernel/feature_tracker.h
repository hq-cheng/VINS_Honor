#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "CameraFactory.h"
#include "PinholeCamera.h"

#include "Parameters.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

// 给定特征点的2D像素系坐标，判断该特征点在图像边界内部，还是在外部
bool inBorder(const cv::Point2f &pt);

// 依次取出 v 中status为1的元素，按序返存回 v 中，并 resize 数组 v 的大小
// 即根据 cv::calcOpticalFlowPyrLK()或cv::goodFeaturesToTrack()返回的 status 剔除掉值为0外点
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    // 核心函数，读取图像并进行特征跟踪处理
    void readImage(const cv::Mat &_img,double _cur_time);

    // 对跟踪到的特征点按照被跟踪的次数从大到小进行排序
    // 取出密集跟踪的点使特征点均匀分布
    void setMask();

    // 向 forw_pts 添加 cv::goodFeaturesToTrack 检测到的新特征点
    // 这些新特征点的id皆初始化为-1，track_cnt皆初始化为1
    void addPoints();

    // 更新特征点的全局id
    // 每一帧图像cv::goodFeaturesToTrack()检测的新特征点id初始值为-1
    // 从readImage()函数读入的第一帧图像开始, 所有特征点id随n_id由0开始累加
    bool updateID(unsigned int i);

    // 创建一个相机模型camodocal::CameraPtr对象, 读取calib_file中的相机相关参数来配置该对象属性
    void readIntrinsicParameter();

    void showUndistortion(const string &name);
    
    // 通过对极约束中的基础矩阵 F 剔除外点
    void rejectWithF();

    // 对特征点进行去畸变校正, 对特征点相机系坐标进行归一化处理, 计算每一个特征点的速度
    // 特征点速度是利用特征点相机系归一化坐标进行计算得到
    void undistortedPoints();

    // setMask()调用, 算法技巧, 设置mask使特征点分布均匀
    cv::Mat mask;
    // 鱼眼相机mask
    cv::Mat fisheye_mask;                             

    // 命名规则:
    // readImage()函数读入的当前帧数据以 forw_xxx 形式命名
    // readImage()函数输出的当前帧数据以 cur_xxx 形式命名
    // readImage()函数末尾会将当前帧数据以 cur_xxx = forw_xxx; 的形式输出
    // readImage()函数末尾会将前一帧数据以 prev_xxx = cur_xxx; 的形式暂存

    // prev_img 只用于暂存前一帧图像的相关数据
    // cur_img 光流跟踪cv::calcOpticalFlowPyrLK()需要的第一帧图像(跟踪之前为前一帧, 跟踪之后为输出的当前帧)
    // forw_img 光流跟踪cv::calcOpticalFlowPyrLK()需要的第二帧图像(跟踪之前为输入的当前帧, 跟踪之后赋值给cur_img输出)
    cv::Mat prev_img, cur_img, forw_img;
    // 暂时存储cv::goodFeaturesToTrack()检测到的新特征点像素系坐标
    vector<cv::Point2f> n_pts;
    // prev_pts 只用于暂存前前一帧图像的特征点像素系坐标(跟踪上的特征点+补充的新特征点)
    // cur_pts 上一帧图像cur_img的特征点像素系坐标(跟踪上的特征点+补充的新特征点)
    // forw_pts 当前帧图像forw_pts的特征点像素系坐标(跟踪上的特征点+补充的新特征点)
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    // prev_un_pts 只用于暂存前一帧图像undistortedPoints()后的特征点相机系归一化坐标
    // cur_un_pts 输出的当前帧图像undistortedPoints()后的特征点相机系归一化坐标
    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    // 输出的当前帧图像的特征点速度
    // 特征点速度是利用特征点相机系归一化坐标进行计算得到
    vector<cv::Point2f> pts_velocity;
    // 存储每一个特征点的id
    vector<int> ids;
    // 存储每一个特征点对应的被跟踪成功的次数
    vector<int> track_cnt;
    // 元素类型为 pair< 特征点id, 特征点相机系归一化坐标 >, undistortedPoints()中用于计算特征点的速度
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    // 相机模型camodocal::CameraPtr类型实例
    camodocal::CameraPtr m_camera;
    // cur_time readImage()函数读入的当前帧图像时间戳
    double cur_time;
    double prev_time;

    // 用于计算所有图像的特征点id, updateID()函数调用
    static int n_id;
};
