#include "initial_alignment.h"


// 利用旋转约束估计陀螺仪偏置
// 已知: 
// SFM中得到了 all_image_frame 中第 i,j 两帧图像时刻IMU坐标系到第 l 帧相机系("世界系")的旋转变换
// IMU数据预积分得到了all_image_frame 中相邻i,j两帧图像间IMU的旋转变换
// 旋转约束: (q_clbj)^(-1) * q_clbi * q_bibj = [1 0]^T
// 预积分的一阶泰勒近似: q_bibj = q_bibj * [ 1 0.5*(J^q_bg)*delta_bg ], (J^q_bg)为预积分雅克比, bg为陀螺仪偏置
// => ( (J^q_bg)^T * (J^q_bg) ) *  delta_bg
//           = 2 * (J^q_bg)^T * [ ( (q_bibj)^(-1)*(q_clbi)^(-1)*q_clbj ) ]_vec, []_vec表示只取虚部
// 即构成 A * x = b 形式的最小二乘问题, LDLT分解 A 和 b, 以得到 x 的近似解
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
        // tmp_A = J^q_bg, 预积分雅克比
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        // tmp_b = (q_bibj)^T * (q_clbi)^T * q_clbj, 只取虚部
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;

    }

    // LDLT分解, 来求解 A * x = b 形式的最小二乘问题
    delta_bg = A.ldlt().solve(b);
    LOGI("gyroscope bias initial calibration: %f, %f, %f", delta_bg(0), delta_bg(1), delta_bg(2));

    // 修改滑动窗口队列各图像时刻的陀螺仪偏置
    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;

    // 陀螺仪的偏置发生改变, 重新计算IMU预积分
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}

// 已知模长 G.norm() 是一个非线性约束, 论文将其转换为线性约束: 
// g_cl := G.norm() * g_cl.normalized() + w1*b1 + w2*b2 = G.norm() * g_cl.normalized() + [ b1 b2 ] * [ w1 w2 ]^T
// 该函数用于求解 3*2 大小的矩阵: [ b1 b2 ]
MatrixXd TangentBasis(Vector3d &g0)
{
    // 这里 g0 = g.normalized() * G.norm()
    // 已知模长 G.norm(), g 为RefineGravity() 上一步求解得到的 g_cl
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    // b1
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    // b2
    c = a.cross(b);
    MatrixXd bc(3, 2);
    // [ b1 b2 ]
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

// 根据重力加速度模长已知(9.8左右)这个约束, 进一步优化重力加速度
// 已知模长 G.norm() 是一个非线性约束, 论文将其转换为线性约束: 
// g_cl := G.norm() * g_cl.normalized() + w1*b1 + w2*b2 = G.norm() * g_cl.normalized() + [ b1 b2 ] * [ w1 w2 ]^T
// 其中 g_cl 为 RefineGravity() 上一步求解得到; b1, b2是张成 g_cl 正切空间的两个正交单位向量, [ b1 b2 ] 为 3*2 矩阵;
// 该约束代入 LinearAlignment() 的 A * x = b 中, 求解该超定方程(g_cl的未知数由三个变成w1,w2两个), 得到新的 g_cl
void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    // 重复迭代, 直至收敛得到新的 g_cl
    for(int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            //       _                                                                          _
            //      |                                                                            |
            // A =  |   -dt  0                0.5*R_bicl*dt*dt*lxly   R_bicl*( p_clcj - p_clci ) |
            //      |   -I   R_bicl*R_clbj    R_bicl*dt*lxly          0                          |
            //      |_                                                                          _|
            //       _                                                                                  _ 
            //      |                                                                                    |
            // b =  |  alpha_bibj - p_bc + R_bicl*R_clbj*p_bc - 0.5*R_bicl*dt*dt*g.normalized()*G.norm() |
            //      |               beta_bibj - R_bicl*dt*g.normalized()*G.norm()                        |
            //      |_                                                                                  _|
            // TangentBasis()函数用于求解 3*2 大小的矩阵: lxly = [ b1 b2 ]
            // 已知模长 G.norm(), g 为RefineGravity() 上一步求解得到的 g_cl
            double dt = frame_j->second.pre_integration->sum_dt;
            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b);

            // 线性约束: g_cl := G.norm() * g_cl.normalized() + [ b1 b2 ] * [ w1 w2 ]^T
            // 每次估计完 g_cl 以后, 都会归一化使其模值接近 G.norm(), w1,w2通常很小(保证g_cl模值不大于G.norm())
            VectorXd dg = x.segment<2>(n_state - 3);
            g0 = (g0 + lxly * dg).normalized() * G.norm();
            //double s = x(n_state - 1);
    }   
    g = g0;
}

// 利用平移约束估计尺度, 重力加速度和速度
// 预积分量约束, 其中 v_wi, v_wj 表示 i, j 图像时刻IMU坐标系的速度在"世界系"下的表示
// alpha_bibj = q_biw * ( p_wbj - p_wbi - v_wi*dt + 0.5*g_w*dt*dt )
//  beta_bibj = q_biw * ( v_wj - v_wi + g_w*dt )
// 将第 l 帧相机系看作"世界系", 其中 v_bi, v_bj 表示 i, j 图像时刻IMU坐标系的速度在IMU系下的表示
// => alpha_bibj = R_bicl * ( s*p_clbj - s*p_clbi - R_clbi*v_bi*dt + 0.5*g_cl*dt*dt )
//     beta_bibj = R_bicl * ( R_clbj*v_bj - R_clbi*v_bi + g_cl*dt )
// 视觉SFM得到的 p_clbi, p_clbj 尺度不确定, 即非米制单位, 乘以尺度 s 缩放为米制单位
// 代入平移约束 s * p_clbi = s * p_clci - R_clbi * p_bc
// => alpha_bibj = R_bicl * ( s*p_clcj - s*p_clci  - R_clbj*p_bc + R_clbi*p_bc ) 
//               + R_bicl * ( - R_clbi*v_bi*dt + 0.5*g_cl*dt*dt )
// => alpha_bibj = s*R_bicl*( p_clcj - p_clci ) - R_bicl*R_clbj*p_bc + p_bc - v_bi*dt + 0.5*R_bicl*g_cl*dt*dt
//     beta_bibj = R_bicl*R_clbj*v_bj - v_bi + R_bicl*g_cl*dt
// =>  _                                       _ 
//    |                                         |
//    |  alpha_bibj - p_bc + R_bicl*R_clbj*p_bc |
//    |               beta_bibj                 |
//    |_                                       _|
//                  _                                                                                      _
//                 |                                                                                        |
//               = |   -dt*v_bi +             0*v_bj + 0.5*R_bicl*dt*dt*g_cl + R_bicl*( p_clcj - p_clci )*s |
//                 |      -v_bi + R_bicl*R_clbj*v_bj +        R_bicl*dt*g_cl +                          0*s |
//                 |_                                                                                      _|
// 待估计量: [ v_bi, v_bj, ..., v_bN g_cl s ]^T, 向量长度: N*3+3+1, 需估计每一帧在IMU系下的速度, 故 N = all_image_frame.size()
// 已知: IMU预积分量 alpha_bibj, beta_bibj; 视觉SFM结果 R_bicl, , p_clci, R_clbj, p_clcj; 外参: p_bc(直接测量, 米制单位)
// 构建 A * x = b 形式的最小二乘问题, LDLT分解 A 和 b, 以得到 x 的近似解
bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    // 待估计量的向量长度
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 10); // 10 = 3(v_bi)+3(v_bj)+3(g_cl)+1(s)
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        //       _                                                                         _
        //      |                                                                           |
        // A =  |   -dt  0                    0.5*R_bicl*dt*dt   R_bicl*( p_clcj - p_clci ) |
        //      |   -I   R_bicl*R_clbj        R_bicl*dt          0                          |
        //      |_                                                                         _|
        //       _                                       _ 
        //      |                                         |
        // b =  |  alpha_bibj - p_bc + R_bicl*R_clbj*p_bc |
        //      |               beta_bibj                 |
        //      |_                                       _|
        double dt = frame_j->second.pre_integration->sum_dt;
        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        // 0.5*R_bicl*dt*dt
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        // R_bicl*( p_clcj - p_clci )
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
        // alpha_bibj - p_bc + R_bicl*R_clbj*p_bc 
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        // R_bicl*R_clbj
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        // R_bicl*dt
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        // beta_bibj
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        // 每一帧图像都有一组 tmp_A * tmp_x = tmp_b, 其中 tmp_A 6*10, tmp_x 10*1(表示这一帧待估计量), tmp_b 6*1
        // 除第一帧, 每一组6个等式, 约束方程共6(N−1)个, 总的待估计量是3N+3+1个, 当N>2, 3N+3+1<6(N−1), 即 A * x = b 是超定方程
        // 超定方程无解, 但可求其最小二乘解，即等式左右两端乘上A的转置矩阵, A := A^T * A, b := A^T * b, 然后求解 A * x = b 方程
        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    
    // ???乘以1000是LDLT分解中的数值技巧, 各乘以1000, 避免数据过小, 计算过程出现截断误差
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);

    double s = x(n_state - 1) / 100.0;
    LOGI("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4);
    LOGI("result g norm: %f, and g: %f, %f, %f", g.norm(), g(0), g(1), g(2));
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0)
    {
        return false;
    }

    // 根据重力加速度模长已知(9.8)这个约束, 进一步优化重力加速度, 同时优化其它待估计量
    RefineGravity(all_image_frame, g, x);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    LOGI("refine g norm: %f, and g: %f, %f, %f", g.norm(), g(0), g(1), g(2));
    if(s < 0.0 )
        return false;   
    else
        return true;
}

// solveGyroscopeBias()函数利用旋转约束估计陀螺仪偏置, LinearAlignment()函数利用平移约束估计尺度, 重力加速度和速度
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    solveGyroscopeBias(all_image_frame, Bgs);

    if(LinearAlignment(all_image_frame, g, x))
        return true;
    else 
        return false;
}
