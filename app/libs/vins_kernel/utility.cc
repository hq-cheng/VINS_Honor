#include "utility.h"

// 参考论文 Visual-Inertial Monocular SLAM with Map Reuse
// g 为 LinearAlignment() 函数返回的第 l 帧图像相机系下的重力加速度 g_cl
// z 轴方向向量应该属于惯性世界坐标系, g2R() 函数主要是保证重力加速度 g_cl 变换后与惯性世界坐标系下重力加速度方向一致
// yaw 角是不可观的, 所以我们希望它初始时设置为 0, 当然 g2R() 函数这一步只是一个小技巧, 为下面得到最终的 R0 做准备
Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    // LinearAlignment() 函数返回的第 l 帧图像相机系下的重力加速度 g_cl, 得到重力加速度 g_cl 方向向量
    Eigen::Vector3d ng1 = g.normalized();
    // 惯性世界坐标系下z轴方向向量, 我们最终希望重力加速度方向对齐到惯性世界坐标系下
    Eigen::Vector3d ng2{0, 0, 1.0};
    // FromTwoVectors() 函数返回 ng1, ng2 两向量间的单位四元数, 即旋转矩阵
    // 得到重力加速度 g_cl 方向向量 ng1 与 z 轴方向向量 ng2 的旋转矩阵
    // 即 R0 = Rot_z(y1)*Rot_y(p1)*Rot_x(r1), 此时 ng1 = Rot_x(-r1)*Rot_y(-p1)*Rot_z(-y1) * ng2
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();

    // 一个小技巧, 设置 yaw 角为 0, 即 R0 = Rot_z(-y1) * R0 = Rot_y(p1)*Rot_x(r1), 最后 g2R() 函数返回此 R0 矩阵
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}
