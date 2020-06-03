#pragma once 
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <android/log.h>
#define LOG_TAG_INITIALSFM "initial_sfm.cc"
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO, LOG_TAG_INITIALSFM, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG_INITIALSFM, __VA_ARGS__)

using namespace Eigen;
using namespace std;


// 存储特征点相关的属性
// VIO初始化函数 initialStructure() 中会将 f_manager.feature 中的每一个特征点属性等信息存储到 vector<SFMFeature> 数组 sfm_f 中
struct SFMFeature
{
    bool state; 		// 特征点是否已被三角化处理
    int id;				// 特征点全局ID
    // pair< frame_count, point >, 即该特征点被滑动窗口队列中第 frame_count 帧图像观测到
	// point 为该特征点投影到这帧图像相机系归一化平面上的坐标
	vector<pair<int,Vector2d>> observation;	
    double position[3]; // 三角化处理得到的特征点3D坐标, 参考 GlobalSFM::triangulatePoint() 函数
    double depth;	
};

// 用于 Vision-only SFM 时构建纯视觉的重投影误差
struct ReprojectionError3D
{
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v)
		{}

	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p);
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];
		T xp = p[0] / p[2];
    	T yp = p[1] / p[2];
    	residuals[0] = xp - T(observed_u);
    	residuals[1] = yp - T(observed_v);
    	return true;
	}

	// 构建残差
	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          ReprojectionError3D, 2, 4, 3, 3>(
	          	new ReprojectionError3D(observed_x,observed_y)));
	}

	double observed_u;
	double observed_v;
};

// Vision-only SFM具体实现类
// VIO初始化函数 initialStructure() 中实现对滑动窗口队列中每一帧图像进行SFM问题求解
class GlobalSFM
{
public:
	GlobalSFM();

	// 以第 l 帧图像为参考坐标系, 即"世界系"
    // 已知当前帧(滑窗最新一帧)与第l帧之间的相对变换R, t(当前帧相对于第l帧), 先进行三角化处理得到一些特征点在"世界系"下的3D位置
    // 利用已知的3D-2D信息求解滑动窗口内其它图像相对于"世界系"的位姿, 并三角化处理得到更多特征点在"世界系"下的3D位置
    // 采用ceres对滑动窗口队列中所有的图像进行纯视觉BA优化, 将优化后的位姿(相对于"世界系")存储到数组 Q, T中并返回
    // 将所有特征点经三角化处理成功得到的"世界系"下3D位置存储到数组 sfm_tracked_points 中并返回
	bool construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

private:
	
	// 已知特征点相对于第 l 帧的3D坐标, 以及其投影到第 i 帧图像的2D位置(相机系归一化平面)
	// 调用 cv::solvePnP() 进行PnP求解, 得到"世界系"(以第 l 帧相机系为参考坐标系)到当前第 i 帧的投影关系R, t
	bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);

	// 视觉特征三角化处理, DLT直接线性变换法, 需要这两帧图像的投影矩阵 Pose0, Pose1
	// 三角化时需要各帧图像的投影矩阵, 相当于"世界系"到各帧图像的相机系, 而这里以第 l 帧图像相机系为参考坐标系, 即"世界系"
	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
							Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
	
	// 对第 frame0 帧图像和第 frame1 帧图像共同观测到的特征点进行三角化处理
	void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
							  int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
							  vector<SFMFeature> &sfm_f);

	int feature_num;	// 特征点的总个数
};