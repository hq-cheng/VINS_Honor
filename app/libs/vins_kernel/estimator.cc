#include "estimator.h"

Estimator::Estimator(): f_manager{Rs}
{
    LOGI("init begins");
    clearState();
}

// 设置在节点/vins_estimator中需要用到的参数
void Estimator::setParameter()
{
    // 第i个相机与IMU之间的外参
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i]; // Rotation from camera frame to imu frame
    }
    // 设置特征点管理器中的外参ric
    f_manager.setRic(ric);
    // 视觉约束项信息矩阵相关参数
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    // time offset 的初始值
    td = TD;
}

// 清空或初始化滑动窗口队列中所有的状态量，外参，状态估计器，feature manager等
void Estimator::clearState()
{
    // 清空或初始化滑动窗口队列中的PVQ, Bais, 预积分量
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    // 重置IMU与相机之间的外参
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    // 重置状态估计器
    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;


    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    // 重置 f_manager
    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

/**
 * IMU: ... t0(i) t2 t3 t4 t5(j) ...
 * IMG: ...                t5(j) ... 
 * getMeasurements()中获取上面格式的IMU和图像数据(时间戳对齐)后, 使用了一个for循环遍历这一段IMU数据(i~j时刻)
 * 并调用processIMU()函数，进行IMU预积分和运动模型更新
 * 递推求解这一段时间末尾时刻（即j图像时刻）的预积分状态量, 以及运动模型更新预测得到的状态量
 */ 
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    // 暂存第一帧IMU数据, 用于中值积分
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    // 刚开始需要先填充滑动窗口队列内的数据，frame_count初始为0，处理一帧图像后加1，直至滑动窗口队列图像的数量为(WINDOW_SIZE + 1)
    // 数组 pre_integrations 的大小为(WINDOW_SIZE + 1)，初始时为空，填充完后包含滑动窗口队列内(WINDOW_SIZE + 1)个预积分状态量
    if (!pre_integrations[frame_count])
    {
        // 与滑动窗口队列同步，填充完数组 pre_integrations 中的预积分状态量
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }

    // 初始第一帧图像对应的 frame_count=0，该图像帧对应的IMU数据不会进行预积分处理，只会记录对应的加速度和角速度初始值
    // 而第二帧图像 frame_count=1 时，用第一、二帧图像间的IMU数据得到第二帧图像时刻的预积分状态量, 以及运动模型更新预测得到的状态量
    if (frame_count != 0)
    {
        // push_back() 函数中会调用IMU预积分更新函数 propagate()
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        // 运动模型更新, 采用中值积分递推公式离散化运动模型
        // 利用[k-1, k]IMU时刻的两帧IMU数据, 预测当前k时刻IMU的PVQ
        // 一段IMU数据最后递推求解得到这一段时间末尾时刻（即j图像时刻）的PVQ
        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/**
 * IMU: ... t0(i) t2 t3 t4 t5(j) ...
 * IMG: ...                t5(j) ... 
 * getMeasurements()中获取上面格式的IMU和图像数据(时间戳对齐)后, 遍历j时刻图像的每一个特征点
 * 将这些特征点信息(特征点像素坐标, 速度, 相机系归一化平面坐标)封装为map数组image
 * !!!VIO后端入口函数, 用于VIO初始化, 视觉特征三角化, 后端BA非线性优化
 */ 
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double &header)
{
    LOGI("new image coming ------------------------------------------");
    LOGI("Adding feature points %u", image.size());

    // 在插入当前帧图像之前, 判断其前两帧图像(2nd last and 3rd last)之间的视差，即(相机系归一化平面下)特征点在这两帧中运动了多少个单位
    // 如果视差大于设定的阈值，则这一帧图像被判定为新的关键帧，而把滑动窗口队列中最老的一帧图像给marg掉
    // 如果视差小于设定的阈值，即前两帧图像之间没怎么运动，则把滑动窗口队列中倒数第二帧图像(2nd last)给丢掉，同时预积分和运动模型积分顺延
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_SECOND_NEW;

    LOGI("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    LOGI("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    LOGI("Solving %d", frame_count);
    LOGI("number of feature: %d", f_manager.getFeatureCount());
    
    // 将读入的这一帧图像, 时间戳, 临时预积分状态量保存到对应数据结构之中
    Headers[frame_count] = header;
    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header, imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    // ESTIMATE_EXTRINSIC = 0 表示IMU与相机之间的外参已是准确值，不需要再优化
    // ESTIMATE_EXTRINSIC = 1 表示IMU与相机之间的外参只是一个估计值，需要将其作为初始值放入BA中进行非线性优化
    // ESTIMATE_EXTRINSIC = 2 表示IMU与相机之间的外参不确定，需要先完成标定，主要是标定外参的旋转矩阵 ric
    if(ESTIMATE_EXTRINSIC == 2)
    {
        LOGI("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            // 取出当前帧图像与其前一帧图像共同观测到的所有特征点相机系归一化平面坐标, 以组合数组 corres 形式返回
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            // CalibrationExRotation() 在线进行IMU与相机之间的外参标定
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                LOGI("initial extrinsic rotation calib success");
                LOGI("initial extrinsic rotation: \n %f, %f, %f, \n %f, %f, %f, \n %f, %f, %f.", calib_ric(0, 0), calib_ric(0, 1), calib_ric(0, 2),
                                                                                                 calib_ric(1, 0), calib_ric(1, 1), calib_ric(1, 2),
                                                                                                 calib_ric(2, 0), calib_ric(2, 1), calib_ric(2, 2));
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    // IMU与相机之间的外参确定后，对VIO进行初始化
    if (solver_flag == INITIAL)
    {
        // 初始时，frame_count=0
        // 当frame_count < WINDOW_SIZE时，只是将图像数据放入滑动窗口队列(同时进行IMU预积分和运动模型更新照常进行)，frame_count++
        // 当frame_count = WINDOW_SIZE时，即滑动窗口队列内图像数量为(WINDOW_SIZE + 1)后，才会将IMU与图像数据一起进行BA非线性优化
        if (frame_count == WINDOW_SIZE)
        {
            // 确保有足够的图像数据参与VIO初始化
            bool result = false;
            if( ESTIMATE_EXTRINSIC != 2 && (header- initial_timestamp) > 0.1)
            {
               // VIO初始化
               result = initialStructure();
               initial_timestamp = header;
            }
            // VIO初始化成功，solver_flag = NON_LINEAR，进行一次BA非线性优化
            if(result)
            {
                solver_flag = NON_LINEAR;
                solveOdometry();
                slideWindow();
                // 移除掉 f_manager.feature 数组中 solve_flag = 2的特征点, 即深度值为负数的特征点
                f_manager.removeFailures();
                LOGI("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
                
            }
            // VIO初始化失败，执行滑窗操作，获取新的关键帧，准备再一次初始化，直至成功
            // initialStructure()失败时返回的 marginalization_flag 可以是 MARGIN_OLD, 也可能是 MARGIN_SECOND_NEW
            else
                slideWindow();
        }
        // 刚开始需要先填充滑动窗口队列内的数据，frame_count初始为0，处理一帧图像后加1，直至滑动窗口队列图像的数量为(WINDOW_SIZE + 1)
        else
            frame_count++;
    }
    // solver_flag = NON_LINEAR，BA非线性优化
    else
    {
        // 核心内容，BA非线性优化
        solveOdometry();

        // 故障检测, 检验 solveOdometry() 优化效果
        // 故障: 这帧图像时刻 IMU 加速度/角速度 bias 估计值过大, 这帧图像位姿估计值相较于其前一帧图像变化过大
        if (failureDetection())
        {
            LOGI("failure detection!");
            failure_occur = 1;
            // 检测到故障，估计器重置，solver_flag = INITIAL
            clearState();
            setParameter();
            LOGI("system reboot!");
            return;
        }

        // 滑窗操作，决定marg/保留滑动窗口队列中的关键帧
        slideWindow();
        // 移除掉 f_manager.feature 数组中 solve_flag = 2的特征点, 即深度值为负数的特征点
        f_manager.removeFailures();
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

// VINS-Mono 采用了视觉和IMU的松耦合初始化方案
// 1 Vision-only SFM: 用从运动恢复结构(SFM)得到纯视觉系统的初始化, 即滑动窗口中所有帧的位姿和所有路标点的3D位置
// 2 Visual Inertial Alignment: 将 Vision-only SFM 结果与IMU预积分结果进行对齐, 恢复尺度因子、重力、陀螺仪偏置和每一帧在IMU系下的速度

// VIO初始化(单目视觉没有尺度信息, IMU测量又存在偏置误差)
// VIO初始化时, frame_count == WINDOW_SIZE, 确保有足够的图像数据参与VIO初始化
bool Estimator::initialStructure()
{

    // check imu observibility
    // 遍历滑动窗口队列中的每一帧图像, 计算线加速度的标准差, 判断IMU是否有充分的运动激励，来进行VIO初始化
    // 充分的运动激励使得尺度可观, 例如匀速运动，IMU加速度计只能测到重力, 那么视觉SLAM的尺度就求不出来
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            // dt 两帧图像时刻之间的时间间隔
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        // 这里并没有算上 all_image_frame 的第一帧，所以求均值和标准差的时候要减1
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        LOGI("IMU variation %f!", var);
        if(var < 0.25)
        {
            LOGI("IMU excitation not enouth!");
            //return false;
        }
    }

    // global sfm
    // frame_count = WINDOW_SIZE, 当前滑动窗口队列中最新一帧图像序号即为 WINDOW_SIZE
    // 滑动窗口队列中共有 frame_count + 1 = WINDOW_SIZE + 1 帧图像
    Quaterniond Q[frame_count + 1];        
    Vector3d T[frame_count + 1];           
    map<int, Vector3d> sfm_tracked_points;
    // 将 f_manager.feature 中的每一个特征点属性等信息存储到 vector<SFMFeature> 数组 sfm_f 中
    vector<SFMFeature> sfm_f;
    // f_manager.feature, list<FeaturePerId>类型数组, 存储所有特征点的属性(FeaturePerId), 及其对应的帧属性
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }

    // 找到滑动窗口队列中从第一帧开始, 最先与最新一帧图像共同观测到的特征点个数大于20
    // 且两图像间(平均视差*焦距)大于30的那一帧图像序号 l
    // 并返回当前帧与第l帧之间的相对变换R, t(当前帧相对于第l帧)
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l))
    {
        // 没找到满足条件的图像
        LOGI("Not enough features or parallax; Move device around");
        // 此时的 marginalization_flag 可以是 MARGIN_OLD, 也可能是 MARGIN_SECOND_NEW
        return false;
    }

    // 以第 l 帧图像为参考坐标系, 即"世界系"
    // 已知当前帧(滑窗最新一帧)与第l帧之间的相对变换R, t(当前帧相对于第l帧), 先进行三角化处理得到一些特征点在"世界系"下的3D位置
    // 利用已知的3D-2D信息求解滑动窗口内其它图像相对于"世界系"的位姿, 并三角化处理得到更多特征点在"世界系"下的3D位置
    // 采用ceres对滑动窗口队列中所有的图像进行纯视觉BA优化, 将优化后的位姿(相对于"世界系")存储到数组 Q, T中并返回
    // 将所有特征点经三角化处理成功得到的"世界系"下3D位置存储到数组 sfm_tracked_points 中并返回
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        LOGI("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // solve pnp for all frame
    // 对于所有的图像，包括不在滑动窗口队列中的，提供初始的R, t估计值，然后solvePnP进行优化
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        // Headers数组大小为 WINDOW_SIZE + 1, 存储滑动窗口队列内图像的时间戳, 参考: processImage()中的 Headers[frame_count]
        // 当这帧图像 frame_it 恰好是滑动窗口队列中的图像时, 只需要存储前面SFM的结果即可
        if((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            // 存储这帧图像时刻IMU坐标系到第 l 帧相机系("世界系")的旋转变换, 以及这帧图像到第 l 帧相机系的平移向量
            // 后面Visual Inertial Alignment会需要, 参考论文 V-B Visual-Inertial Alignment 的公式
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        // 有点像kmp算法的2把尺子，大尺子是 all_image_frame(所有图像), 小尺子是 Headers(滑动窗口队列)
        // VIO初始化时, frame_count == WINDOW_SIZE, 即 Headers 滑动窗口队列在初始化前就已经填充完毕
        // all_image_frame 按照时间戳排序, 由于 MARGIN_OLD/MARGIN_SECOND_NEW, 两数组尾部的时间戳一致
        // 例如, Headers 从(c0 c1 c2 c3 c4)开始, 依次marg掉图像 c3 , c4, c5, c0, c7, c1得到下面的结果
        // all_image_frame: c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10
        //         Headers:                   c2 c6 c8 c9 c10
        // i 作为 Headers 的索引只有 (frame_it->first) >= Headers[i] 时才会增加
        if((frame_it->first) > Headers[i])
        {
            i++;
        }

        // 当 (frame_it->first) < Headers[i] 时, 即滑动窗口队列外的图像, 非关键帧, 没有参与SFM
        // 存储这些图像时刻IMU坐标系到第 l 帧相机系("世界系")的旋转变换, 以及这些图像到第 l 帧相机系的平移向量
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            //LOGI( "pts_3_vector size: %u!", pts_3_vector.size() );
            LOGI("Not enough points for solve pnp !");
            return false;
        }
        // 已知特征点相对于第 l 帧的3D坐标, 以及其投影到这帧图像的2D位置(相机系归一化平面)
        // 调用 cv::solvePnP() 进行PnP求解, 得到"世界系"(以第 l 帧相机系为参考坐标系)到这一帧图像的投影关系R, t
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            LOGI("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose(); // 取逆得到这一帧图像相对于"世界系"(以第 l 帧相机系为参考坐标系)的变换关系
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        // 存储这些图像时刻IMU坐标系到第 l 帧相机系("世界系")的旋转变换, 以及这些图像到第 l 帧相机系的平移向量
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }

    // Visual Inertial Alignment: 将 Vision-only SFM 结果与IMU预积分结果进行对齐, 恢复尺度因子、重力、陀螺仪偏置和每一帧在IMU系下的速度
    if (visualInitialAlign())
        return true;
    else
    {
        LOGI("misalign visual structure with IMU");
        return false;
    }

}

// Visual Inertial Alignment: 将 Vision-only SFM 结果与IMU预积分结果进行对齐, 恢复尺度因子、重力、陀螺仪偏置和每一帧在IMU系下的速度
// 陀螺仪的偏置校准(加速度偏置没有处理)
// 初始化速度、重力、尺度因子
// 更新了陀螺仪偏置后, IMU数据需要repropagate;
// 得到尺度和重力的方向后, 需更新滑动窗口队列中所有图像在"世界系"下的PVQ
bool Estimator::visualInitialAlign()
{
    VectorXd x;
    // solveGyroscopeBias()函数利用旋转约束估计陀螺仪偏置, LinearAlignment()函数利用平移约束估计尺度, 重力加速度和速度
    // 陀螺仪的偏置发生改变, solveGyroscopeBias()会重新计算预积分
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        LOGI("solve g failed!");
        return false;
    }

    // 获取滑动窗口队列中所有图像的位姿Ps、Rs, 并将其置为关键帧
    for (int i = 0; i <= frame_count; i++)
    {
        // 图像 i 时刻IMU坐标系到第 l 帧相机系("世界系")的旋转变换, 以及这些图像到第 l 帧相机系的平移向量
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    // 重新进行三角化处理, 计算所有特征点的深度
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    // triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    
    // 陀螺仪的偏置发生改变, 重新计算预积分
    // ???solveGyroscopeBias()函数中陀螺仪偏置改变后, 也重新计算预积分, 是否有点重复, 或用于进一步提升精度
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }

    // 根据尺度 s 重置 Ps, Vs 和 depth
    for (int i = frame_count; i >= 0; i--)
        // R_clb0*p_b0bi = ( s*p_clci - R_clbi*p_bici ) - ( s*p_clc0 -R_clb0*p_b0c0 ) = s*p_clbi - s*p_clb0
        // 位移设置为: 滑动窗口队列中, 第 0 帧图像时刻IMU系与第 i 帧图像时刻IMU系之间的相对位移, 仍以第 l 帧图像相机系为参考坐标系
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            // v_cli = R_clbi * v_bi
            // 其中 v_wi 为 i 图像时刻IMU坐标系的速度在第l帧图像相机系下的表示, v_bi 为 i 图像时刻IMU坐标系的速度在IMU系下的表示
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    // f_manager.feature, list<FeaturePerId>类型数组, 存储所有特征点的属性(FeaturePerId), 及其对应的帧属性
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // 如果该特征点没有被两帧或两帧以上且都在滑动窗口队列内的图像同时观测到, 该特征点没有意义
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        // 深度值按尺度 s 缩放, 恢复到米制单位
        it_per_id.estimated_depth *= s;
    }

    
    
    // g2R() 函数中, 首先得到重力加速度 g_cl 方向向量 ng1 与 z 轴方向向量 ng2 的旋转矩阵
    // 即 R0 = Rot_z(y1)*Rot_y(p1)*Rot_x(r1), 此时 ng1 = Rot_x(-r1)*Rot_y(-p1)*Rot_z(-y1) * ng2
    // 接着, 一个小技巧, 设置 yaw 角为 0, 即 R0 = Rot_z(-y1) * R0 = Rot_y(p1)*Rot_x(r1), 最后 g2R() 函数返回此 R0 矩阵
    // 注: 
    // g 为 LinearAlignment() 函数返回的第 l 帧图像相机系下的重力加速度 g_cl
    // z 轴方向向量应该属于惯性世界坐标系, g2R() 函数主要是保证重力加速度 g_cl 变换后与惯性世界坐标系下重力加速度方向一致
    // yaw 角是不可观的, 所以我们希望它初始时设置为 0, 当然 g2R() 函数这一步只是一个小技巧, 为下面得到最终的 R0 做准备
    Matrix3d R0 = Utility::g2R(g);
    
    // Rs[0] 为第 0 帧图像时刻IMU坐标系到第 l 帧相机系的旋转矩阵, 假设 Rs[0] = Rot_z(ys0)*Rot_y(ps0)*Rot_x(rs0)
    // 假设旋这里旋转矩阵 R0 * Rs[0] = Rot_y(p1)*Rot_x(r1) * Rot_z(ys0)*Rot_y(ps0)*Rot_x(rs0) 的 yaw 角表示为 y2
    // 即亦有 R0 * Rs[0] = Rot_z(y2)*Rot_y(p2)*Rot_x(r2) 这种表示, 后面会用到
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    // 则此时 R0 = Rot_z( -y2 ) * Rot_y(p1)*Rot_x(r1)
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    
    // 由前面 ng1 = Rot_x(-r1)*Rot_y(-p1)*Rot_z(-y1) * ng2, 得到 g_cl = Rot_x(-r1)*Rot_y(-p1)*Rot_z(-y1) * ng2 * g_cl.norm()
    // g = R0 * g_cl = Rot_z( -y2 ) * Rot_y(p1)*Rot_x(r1) * Rot_x(-r1)*Rot_y(-p1)*Rot_z(-y1) * ng2 * g_cl.norm()
    //               = Rot_z( -y2-y1 ) * ng2 * g_cl.norm(), ng2为 z 轴方向向量
    // 对于仍以yaw角满足 Rot_z( y ) * ng2 = ng2, 故最终将 g_cl 与 z 轴方向向量对齐, 显然 yaw 角是不可观的
    g = R0 * g;                   // R_wcl * ( g_cl )

    // 当然你可以理解为 rot_diff = R0 是从第 l 帧图像相机系到惯性世界坐标系的旋转矩阵, 那么可以有下面这样的表述
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        // 惯性世界坐标系, 即与第 0 帧图像时刻的IMU坐标系原点重合的惯性坐标系, 没有相对平移, 故 p_wb0 = 0
        Ps[i] = rot_diff * Ps[i]; // R_wcl * ( R_clb0*p_b0bi ) + p_wb0
        Rs[i] = rot_diff * Rs[i]; // R_wcl * ( R_clbi )
        Vs[i] = rot_diff * Vs[i]; // R_wcl * ( v_cli )
    }

    // 注意前面因为 yaw 角是不可观的所以我们希望它初始时设置为 0, 这体现在 Rs[0] = rot_diff * Rs[0] 后的yaw角上
    // rot_diff * Rs[0] = Rot_z( -y2 ) * Rot_y(p1)*Rot_x(r1) * Rot_z(ys0)*Rot_y(ps0)*Rot_x(rs0)
    // 而前面有 R0 * Rs[0] = Rot_y(p1)*Rot_x(r1) * Rot_z(ys0)*Rot_y(ps0)*Rot_x(rs0), 这里 R0 为g2R() 函数返回
    // 且 R0 * Rs[0] 恰好 yaw 角为 y2, 即亦有 R0 * Rs[0] = Rot_z(y2)*Rot_y(p2)*Rot_x(r2) 这种表示
    // => Rs[0] = rot_diff * Rs[0] = Rot_z( -y2 ) * Rot_z(y2)*Rot_y(p2)*Rot_x(r2) = Rot_y(p2)*Rot_x(r2)
    // 至此, 最终 g 方向与惯性世界坐标系下z轴对齐, Rs[0] 的 yaw 角为 0
    LOGI("g0:     %f, %f, %f", g(0), g(1), g(2));// 近似 [ 0 , 0, 9.81] 的形式
    LOGI( "my R0:   %f， %f, %f", (Utility::R2ypr(Rs[0]).transpose())(0), (Utility::R2ypr(Rs[0]).transpose())(1), (Utility::R2ypr(Rs[0]).transpose())(2) );// 近似 [ 0, xxx, xxx ] 的形式

    return true;
}

// 找到滑动窗口队列中从第一帧开始, 最先与最新一帧图像共同观测到的特征点个数大于20
// 且两图像间(平均视差*焦距)大于30的那一帧图像序号 l
// 并返回当前帧与第l帧之间的相对变换R, t(当前帧相对于第l帧)
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        // 给定两帧图像 frame_count = i, frame_count = WINDOW_SIZE(滑动窗口队列中最新的一帧图像)
        // 取出这两帧图像共同观测到的所有特征点的相机系归一化平面坐标 a, b, 并组合为 vector<pair<a, b>> 形式返回
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            // solveRelativeRT() 通过对极约束计算本质矩阵 E
            // 并从 E 中恢复出当前帧与第l帧之间的相对变换R, t(当前帧相对于第l帧), 并判断内点数目是否足够
            // ???平均视差*焦距 > 30
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                // 返回滑动窗口队列中最先满足条件的那一帧图像的序号
                l = i;
                LOGI("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                // 一旦这一帧与当前帧视差足够大了, 就不再继续找下去了(再找只会和当前帧的视差越来越小)
                return true;
            }
        }
    }
    return false;
}

void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        // 对滑动窗口队列中所有图像观测到的深度未知的特征点进行三角化处理
        f_manager.triangulate(Ps, tic, ric);
        optimization();
    }
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        LOGI("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if(relocalization_info)
    { 
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;   
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());

        relocalization_info = 0;
    }
}

// 故障检测, 检验 solveOdometry() 优化效果
// 故障: 这帧图像时刻 IMU 加速度/角速度 bias 估计值过大, 这帧图像位姿估计值相较于其前一帧图像变化过大
bool Estimator::failureDetection()
{
    // 这一帧图像只有 last_track_num<2 个特征点在之前被其它的图像观测到过
    if (f_manager.last_track_num < 2)
    {
        LOGI(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    // IMU 加速度/角速度 bias 估计值过大
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        LOGI(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        LOGI(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    
    // 这帧图像位姿估计值相较于其前一帧图像变化过大
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        LOGI(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        LOGI(" big z translation");
        return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        LOGI(" big delta_angle ");
        //return true;
    }
    return false;
}


void Estimator::optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            LOGI("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            LOGI("estimate extinsic param");
    }
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        //problem.SetParameterBlockConstant(para_Td[0]);
    }

    vector2double();

    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }
    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
 
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point;
            if (ESTIMATE_TD)
            {
                    ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                     it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                    /*
                    double **para = new double *[5];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    para[4] = para_Td[0];
                    f_td->check(para);
                    */
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    LOGI("visual measurement count: %d", f_m_cnt);

    if(relocalization_info)
    {
        LOGI("set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);
        int retrive_feature_index = 0;
        int feature_index = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            int start = it_per_id.start_frame;
            if(start <= relo_frame_local_index)
            {   
                while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                    
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }     
            }
        }

    }

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOGI("Iterations : %d", static_cast<int>(summary.iterations.size()));

    double2vector();

    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        marginalization_info->preMarginalize();

        marginalization_info->marginalize();

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    assert(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            LOGI("begin marginalization");
            marginalization_info->preMarginalize();

            LOGI("begin marginalization");
            marginalization_info->marginalize();
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
}

void Estimator::slideWindow()
{
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;
 
                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);

            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}

void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    for(int i = 0; i < WINDOW_SIZE; i++)
    {
        if(relo_frame_stamp == Headers[i])
        {
            relo_frame_local_index = i;
            relocalization_info = 1;
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}

