#include "initial_ex_rotation.h"

InitialEXRotation::InitialEXRotation(){
    frame_count = 0;
    Rc.push_back(Matrix3d::Identity());
    Rc_g.push_back(Matrix3d::Identity());
    Rimu.push_back(Matrix3d::Identity());
    ric = Matrix3d::Identity();
}

// 在线进行IMU与相机之间的外参标定
// 相邻两图像时刻 i, j 之间有: IMU旋转预积分量 q_bibj, 视觉测量 q_cicj
// 则有: q_bibj * q_bc = q_bc * q_cicj
// 即: ( [q_bibj]_L - [q_cicj]_R ) * q_bc = A * q_bc = 0, []_L与[]_R为四元数左右乘子
// 利用多个图像时刻的数据即可构建超定方程 A * q_bc = 0, 对A进行SVD分解即可得到外参旋转矩阵 q_bc(即ric)的近似值
bool InitialEXRotation::CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result)
{

    // corres 当前帧图像与其前一帧图像共同观测到的所有特征点相机系归一化平面坐标, 并组合为vector数组形式
    // delta_q_imu 相邻两图像(当前帧与其前一帧)时刻之间的IMU旋转预积分量

    frame_count++;
    // 利用对积几何约束求解前后两帧图像间的相对变换关系 R, t(以第一帧图像相机坐标系为参考坐标系), 并返回 R
    Rc.push_back(solveRelativeR(corres));
    Rimu.push_back(delta_q_imu.toRotationMatrix());
    // ric.inverse() * delta_q_imu * ric <=> q_cibi * q_bibj * q_bjcj = q_cicj
    // 相邻两图像间的相机相对旋转矩阵 R, 由IMU预积分量 Rimu 与 外参 ric 求解得到
    // Rc_g 相当于 Rc 的预测值, 一个由视觉测量得到, 一个由IMU测量得到
    Rc_g.push_back(ric.inverse() * delta_q_imu * ric);

    // 构建超定方程 A * q_bc = 0
    Eigen::MatrixXd A(frame_count * 4, 4);
    A.setZero();
    int sum_ok = 0;
    
    // 在下面代码中, 实际是围绕 q_cb * q_bibj  = q_cicj * q_cb 来构建超定方程
    // 即 ( [q_cicj]_L - [q_bibj]_R ) * q_cb = A * q_cb = 0, L为相机旋转四元数的左乘矩阵, R为IMU旋转四元数的右乘矩阵

    // 遍历多个图像时刻的数据来构建超定方程 A * q_cb = 0
    for (int i = 1; i <= frame_count; i++)
    {
        Quaterniond r1(Rc[i]);
        Quaterniond r2(Rc_g[i]);

        // 添加Huber鲁棒核: w_ij = r_ij > threshold ? threshold/r_ij : 1, r_ij为旋转矩阵 Rc_g 与 Rc 之间的角度误差
        // 为了降低外点的干扰, 防止出现误差非常大的 qbibj 和 qcicj约束导致估计的结果偏差太大
        double angular_distance = 180 / M_PI * r1.angularDistance(r2);
        LOGI("%d %f", i, angular_distance);

        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
        ++sum_ok;
        Matrix4d L, R;

        double w = Quaterniond(Rc[i]).w();
        Vector3d q = Quaterniond(Rc[i]).vec();
        L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;

        Quaterniond R_ij(Rimu[i]);
        w = R_ij.w();
        q = R_ij.vec();
        R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;

        // 鲁棒核以权重的形式添加到超定方程 A * q_cb = 0 = 0 中, 即 A := w * A
        A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R);
    }

    // 对超定方程中的矩阵 A 进行SVD分解, 最小奇异值对应的奇异向量即为外参旋转矩阵 q_cb(即ric.inverse())的近似值
    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    Matrix<double, 4, 1> x = svd.matrixV().col(3);
    Quaterniond estimated_R(x);
    ric = estimated_R.toRotationMatrix().inverse();
    //cout << svd.singularValues().transpose() << endl;
    //cout << ric << endl;

    Vector3d ric_cov;
    ric_cov = svd.singularValues().tail<3>(); // 奇异值从大到小存储
    // 至少迭代计算了 WINDOW_SIZE 次，且SVD分解得到的第二小的奇异值大于 0.25 才认为标定成功
    // 最小的奇异值要足够接近于0, 和第二小之间要有足够差距才行, 这样才算求出了最优解
    if (frame_count >= WINDOW_SIZE && ric_cov(1) > 0.25)
    {
        calib_ric_result = ric;
        return true;
    }
    else
        return false;
}

// 利用对积几何约束求解前后两帧图像间的相对变换关系 R, t(以第一帧图像相机坐标系为参考坐标系)
Matrix3d InitialEXRotation::solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres)
{
    if (corres.size() >= 9)
    {
        // corres 当前帧图像与其前一帧图像共同观测到的所有特征点相机系归一化平面坐标, 并组合为vector数组形式
        // 将 corres 中对应特征点的相机系归一化平面坐标放入vector数组 ll, 与 rr 中
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }

        // cv::findFundamentalMat() 利用对极约束求解本质矩阵 E
        cv::Mat E = cv::findFundamentalMat(ll, rr); // 用于求解基础矩阵, 当然也可用于求解基础矩阵
        cv::Mat_<double> R1, R2, t1, t2;
        // 对本质矩阵 E 进行SVD分解, 得到前后两帧图像间4组可能的相对变换关系 R, t
        decomposeE(E, R1, R2, t1, t2);

        if (determinant(R1) + 1.0 < 1e-09)
        {
            E = -E;
            decomposeE(E, R1, R2, t1, t2);
        }

        // 采用三角化对本质矩阵 E 分解得到的每组 R, t 进行验证，选择是特征点深度值为正的的 R, t
        // 返回使用该 R, t 三角化后, 所有特征点中深度值为正值的个数比例
        double ratio1 = max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
        double ratio2 = max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;

        // cv::findFundamentalMat() 得到的 R 是前一帧到当前帧的旋转矩阵
        // OpenCV视觉几何中习惯将前一帧相机系作为参考坐标系, 即 "世界坐标系", 求解前一帧到当前帧的旋转矩阵 "Rcw"
        // 对其求转置则得到当前帧相对于前一帧的旋转矩阵( 即呼应上面的注释, 运动学更习惯于使用 "Rwc" )
        Matrix3d ans_R_eigen;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                ans_R_eigen(j, i) = ans_R_cv(i, j);
        return ans_R_eigen;
    }
    return Matrix3d::Identity();
}

// 采用三角化对本质矩阵 E 分解得到的每组 R, t 进行验证，选择是特征点深度值为正的的 R, t
double InitialEXRotation::testTriangulation(const vector<cv::Point2f> &l,
                                          const vector<cv::Point2f> &r,
                                          cv::Mat_<double> R, cv::Mat_<double> t)
{
    cv::Mat pointcloud;

    // cv::findFundamentalMat() 得到的 R 是前一帧到当前帧的旋转矩阵
    // OpenCV视觉几何中习惯将前一帧相机系作为参考坐标系, 即 "世界坐标系", 求解前一帧到当前帧的旋转矩阵 "Rcw"

    // 第一帧图像的投影矩阵(以第一帧图像相机坐标系为参考坐标系)
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);
    
    // 第二帧图像的投影矩阵
    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2));
    // cv::triangulatePoints() 三角测量函数, 
    // vector<cv::Point2f> l 第一帧图像对应的特征点相机系归一化平面坐标
    // vector<cv::Point2f> r 第二帧图像对应的特征点相机系归一化平面坐标
    // 输出的是三角化后特征点的3D齐次坐标, 参考坐标系为 "世界坐标系"(显然与这两帧图像中前一帧的相机系一致)
    cv::triangulatePoints(P, P1, l, r, pointcloud);
    int front_count = 0;
    for (int i = 0; i < pointcloud.cols; i++)
    {
        double normal_factor = pointcloud.col(i).at<float>(3);
        // 输出的是三角化后特征点的3D齐次坐标, 因此需要将前三个维度除以第四个维度以得到非齐次坐标
        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
            front_count++;
    }
    LOGI("MotionEstimator: %f", 1.0 * front_count / pointcloud.cols);
    // 返回使用该 R, t 三角化后, 所有特征点中深度值为正值的个数比例
    return 1.0 * front_count / pointcloud.cols;
}

// 对本质矩阵 E 进行SVD分解, 得到前后两帧图像间4组可能的变换关系 R, t
void InitialEXRotation::decomposeE(cv::Mat E,
                                 cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                 cv::Mat_<double> &t1, cv::Mat_<double> &t2)
{
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}
