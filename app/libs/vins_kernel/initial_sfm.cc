#include "initial_sfm.h"

GlobalSFM::GlobalSFM(){}

// 视觉特征三角化处理, DLT直接线性变换法, 需要这两帧图像的投影矩阵 Pose0, Pose1
// 三角化时需要各帧图像的投影矩阵, 相当于"世界系"到各帧图像的相机系, 而这里以第 l 帧图像相机系为参考坐标系, 即"世界系"
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	// 构建超定方程 D * y = 0, 对矩阵 D 进行SVD分解, 得到最小奇异值对应奇异向量(x', y', z', w')即为 y = (xw, yw, zw, 1) 的近似解
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	// 三角化测量结果 xw = x'/w', yw = y'/w', zw = z'/w'("世界系"下特征点深度值)
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

// 已知特征点相对于第 l 帧的3D坐标, 以及其投影到第 i 帧图像的2D位置(相机系归一化平面)
// 调用 cv::solvePnP() 进行PnP求解, 得到"世界系"(以第 l 帧相机系为参考坐标系)到当前第 i 帧的投影关系R, t
bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state != true)
			continue;
		Vector2d point2d;
		// 遍历所有观测到该特征点的图像
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			// 筛选出那些被第 i 帧图像观测到, 且已经三角化处理的特征点
			if (sfm_f[j].observation[k].first == i)
			{
				// 该特征点投影到这帧图像相机系归一化平面上的坐标
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
				// 三角化处理得到的特征点3D坐标, 以第 l 帧图像为参考坐标系, 参考 GlobalSFM::triangulatePoint() 函数
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				// 因为该特征在第 i 帧上只会出现一次, 一旦找到了就没有必要再继续找了
				break;
			}
		}
	}

	// 筛选出满足条件的特征点数量少于10, 那么整个VIO初始化全部失败
	if (int(pts_2_vector.size()) < 15)
	{
		LOGI("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10)
			return false;
	}

	// 已知特征点相对于第 l 帧的3D坐标, 以及其投影到第 i 帧图像的2D位置(相机系归一化平面)
	// 调用 cv::solvePnP() 进行PnP求解, 得到"世界系"(以第 l 帧相机系为参考坐标系)到当前第 i 帧的投影关系R, t
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);
	cv::Rodrigues(tmp_r, rvec); 	// R_initial 对应的旋转向量 rvec
	cv::eigen2cv(P_initial, t);		// P_initial 对应的平移向量 t
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ;
	// ???实际上 cv::solvePnP() 输出 rvec, t, 故读入的 R_initial, P_initial 好像用处不大
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	if(!pnp_succ)
	{
		return false;
	}
	cv::Rodrigues(rvec, r);
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;

}

// 对第 frame0 帧图像和第 frame1 帧图像共同观测到的特征点进行三角化处理
void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);

	// 遍历数组 sfm_f 中的每一个特征点
	for (int j = 0; j < feature_num; j++)
	{
		// 如果该特征点已经被三角化处理过, 则跳过
		if (sfm_f[j].state == true)
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
		// 遍历所有观测到该特征点的图像
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == frame0)
			{
				point0 = sfm_f[j].observation[k].second; // 该特征点投影到这帧图像相机系归一化平面上的坐标
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)
			{
				point1 = sfm_f[j].observation[k].second; // 该特征点投影到这帧图像相机系归一化平面上的坐标
				has_1 = true;
			}
		}
		// 如果该特征点确实被第 frame0 帧图像和第 frame1 帧图像共同观测到, 对该特征点进行三角化处理
		if (has_0 && has_1)
		{
			Vector3d point_3d;
			// 视觉特征三角化处理, 需要这两帧图像的投影矩阵 Pose0, Pose1
			// 三角化时需要各帧图像的投影矩阵, 相当于"世界系"到各帧图像的相机系, 而这里以第 l 帧图像相机系为参考坐标系, 即"世界系"
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
		}							  
	}
}

// q w_R_cam t w_R_cam
// c_rotation cam_R_w 
// c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)

// 以第 l 帧图像为参考坐标系, 即"世界系"
// 已知当前帧(滑窗最新一帧)与第l帧之间的相对变换R, t(当前帧相对于第l帧), 先进行三角化处理得到一些特征点在"世界系"下的3D位置
// 利用已知的3D-2D信息求解滑动窗口内其它图像相对于"世界系"的位姿, 并三角化处理得到更多特征点在"世界系"下的3D位置
// 采用ceres对滑动窗口队列中所有的图像进行纯视觉BA优化, 将优化后的位姿(相对于"世界系")存储到数组 Q, T中并返回
// 将所有特征点经三角化处理成功得到的"世界系"下3D位置存储到数组 sfm_tracked_points 中并返回

// @param[in] frame_num: frame_num = frame_count + 1 = WINDOW_SIZE + 1, 滑动窗口队列中图像的个数
// @param[in] l: 滑动窗口队列中第 l 帧图像的序号, 这里以第 l 帧图像的相机系为参考坐标系
// @param[in] relative_R: 初始化滑动窗口队列中最新一帧图像相对于第 l 帧图像的旋转矩阵
// @param[in] relative_T: 初始化滑动窗口队列中最新一帧图像相对于第 l 帧图像的平移向量
// @param[in] sfm_f: vector<SFMFeature>类型数组, 存储每一个特征点属性等信息
// @param[out] q: SFM结果, 滑动窗口队列中每一帧图像相对于第 l 帧图像的 quaternion
// @param[out] T: SFM结果, 滑动窗口队列中每一帧图像相对于第 l 帧图像的 position
// @param[out] sfm_tracked_point: 优化后的特征点相对于第 l 帧图像的 3D position
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
	feature_num = sfm_f.size();
	
	// 以第 l 帧图像的相机系为参考坐标系, 即"世界系"
	q[l].w() = 1;
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero();
	// 当前滑动窗口队列中最新一帧图像序号即为 frame_count = WINDOW_SIZE = frame_num - 1
	// 初始化滑动窗口队列中最新一帧图像相对于第 l 帧图像的位姿 relative_R, relative_T
	q[frame_num - 1] = q[l] * Quaterniond(relative_R); 
	T[frame_num - 1] = relative_T;

	// 下面的各个 c_xxx 容器(包括 Pose 容器)存储的是第 l 帧图像相对于其它帧图像的变换关系
	// 保存这种相反的旋转平移关系, 是因为后面视觉特征三角化处理时需要各帧图像的投影矩阵
	// 三角化时需要各帧图像的投影矩阵, 相当于"世界系"到各帧图像的相机系, 而这里以第 l 帧图像相机系为参考坐标系, 即"世界系"
	Matrix3d c_Rotation[frame_num];
	Vector3d c_Translation[frame_num];
	Quaterniond c_Quat[frame_num];
	double c_rotation[frame_num][4];
	double c_translation[frame_num][3];
	Eigen::Matrix<double, 3, 4> Pose[frame_num];

	c_Quat[l] = q[l].inverse();
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];

	// q[frame_num - 1] 最新一帧到第l帧 -> 第l帧到最新一帧 c_Quat[frame_num - 1]
	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();	
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	// c_Translation[frame_num - 1] 最新一帧下向量 = -第l帧到最新一帧*第l帧下向量 -(c_Rotation[frame_num - 1] * T[frame_num - 1])
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];


	// 1: trangulate between l ----- frame_num - 1
	// 2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1; 
	// 从滑动窗口队列中第 l 帧图像开始, 遍历之后的每一帧图像, 直至最新一帧图像
	// 首先, 对第 l 帧图像和第 frame_num - 1 帧图像(滑窗最新一帧)共同观测到的特征点进行三角化处理
	// 然后, 对于第 l 帧之后的每一帧图像, 利用已三角化的特征点进行PnP求解, 得到"世界系"(以第 l 帧相机系)到这一帧图像的位姿
	// 最后, 利用这一帧得到的位姿, 再和第 frame_num - 1 帧图像(滑窗最新一帧)进行三角化处理它们共同观测到的特征点
	for (int i = l; i < frame_num - 1 ; i++)
	{
		// solve pnp
		if (i > l)
		{
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];
			// 已知特征点相对于第 l 帧的3D坐标, 以及其投影到第 i 帧图像的2D位置(相机系归一化平面)
			// 调用 cv::solvePnP() 进行PnP求解, 得到"世界系"(以第 l 帧相机系为参考坐标系)到当前第 i 帧的投影关系R, t
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f)) // ???读入的 R_initial, P_initial好像用处不大
				return false;
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

		// triangulate point based on the solve pnp result
		// 对第 i 帧图像和第 frame_num - 1 帧图像(滑窗最新一帧)共同观测到的特征点进行三角化处理
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}

	// 3: triangulate l-----l+1 l+2 ... frame_num -2
	// 之前都是选择一帧图像与第 frame_num - 1 帧图像(滑窗最新一帧)进行三角化处理
	// 这里选择一帧图像与第 l 帧进行三角化处理, 补充经三角化处理后特征点的数量
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
	
	// 4: solve pnp l-1; triangulate l-1 ----- l
	//              l-2              l-2 ----- l
	// 对于第 l 帧之前的每一帧图像, 利用已三角化的特征点进行PnP求解, 得到"世界系"(以第 l 帧相机系为参考坐标系)到这一帧图像的位姿
	// 然后, 利用这一帧得到的位姿, 再和第 l 帧图像进行三角化处理它们共同观测到的特征点
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}

	// 5: triangulate all other points
	// 对剩余仍未进行三角化的特征点, 进行三角化处理
	// 至此得到了滑动窗口队列中每一帧图像相对于第 l 帧的位姿, 以及特征点在"世界系"(以第 l 帧相机系为参考坐标系)下的3D坐标
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
		// 1. 未三角化, 2. 被两帧或两帧以上的图像共同观测到
		if ((int)sfm_f[j].observation.size() >= 2)
		{
			Vector2d point0, point1;
			int frame_0 = sfm_f[j].observation[0].first;
			point0 = sfm_f[j].observation[0].second;
			int frame_1 = sfm_f[j].observation.back().first;
			point1 = sfm_f[j].observation.back().second;
			Vector3d point_3d;
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
		}		
	}

/*
	for (int i = 0; i < frame_num; i++)
	{
		q[i] = c_Rotation[i].transpose();
	}
	for (int i = 0; i < frame_num; i++)
	{
		Vector3d t_tmp;
		t_tmp = -1 * (q[i] * c_Translation[i]);
	}
*/
	// 采用ceres进行全局BA, 纯视觉优化
	ceres::Problem problem;
	// 注意, 因为四元数是四维的, 但是自由度是3维的, 因此需要引入 LocalParameterization
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	for (int i = 0; i < frame_num; i++)
	{
		// double array for ceres
		// 以第 l 帧图像的相机系为参考坐标系, 即"世界系"
		// 下面的各个 c_xxx 容器存储的是第 l 帧图像相对于其它帧图像的变换关系, 相当于"世界系"到各帧图像的相机系
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		// 加入待优化量, 每一帧图像的位姿
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
		problem.AddParameterBlock(c_translation[i], 3);
		// 固定先验值
		// 因为第 l 帧是参考坐标系("世界系"), 第 frame_num - 1 帧图像(滑窗最新一帧)的平移也是先验，如果不固定住，原本可观的量会变的不可观
		if (i == l)
		{
			problem.SetParameterBlockConstant(c_rotation[i]);
		}
		if (i == l || i == frame_num - 1)
		{
			problem.SetParameterBlockConstant(c_translation[i]);
		}
	}

	for (int i = 0; i < feature_num; i++)
	{
		if (sfm_f[i].state != true)
			continue;
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
		{
			int l = sfm_f[i].observation[j].first;
			// 取出该特征点投影到这帧图像相机系归一化平面上的坐标, 用于构建重投影误差
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y());

    		problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l], 
    								sfm_f[i].position);	 
		}

	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2;
	ceres::Solver::Summary summary;
	// 显然这里只是纯视觉BA, 只优化了图像位姿, 对于特征点在"世界系"下的3D位置并未优化
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
		LOGI("vision only BA converge");
	}
	else
	{
		LOGE("vision only BA not converge ");
		return false;
	}
	
	for (int i = 0; i < frame_num; i++)
	{
		q[i].w() = c_rotation[i][0]; 
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		// 取逆, 得到其它帧图像相机系相对于第 l 帧图像("世界系")的变换关系
		q[i] = q[i].inverse();
	}
	for (int i = 0; i < frame_num; i++)
	{
		// 得到其它帧图像相机系相对于第 l 帧图像("世界系")的变换关系
		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
	}

	// 将所有已经三角化的特征点存储到 sfm_tracked_points 数组
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
	return true;

}

