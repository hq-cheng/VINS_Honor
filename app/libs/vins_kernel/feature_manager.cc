#include "feature_manager.h"

// 返回观测到该特征点的最后一帧图像序号
int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    // 特征点管理器中的外参ric初始化为单位矩阵, Estimator::setParameter()函数中调用 setRic() 函数修改
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

// 特征点管理器中的外参ric初始化为单位矩阵, Estimator::setParameter()函数中调用 setRic() 函数修改
void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

// 获取 feature 数组中被两帧或两帧以上且都在滑动窗口队列内的图像同时观测到的特征点总个数
int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    // feature, list<FeaturePerId>类型数组, 存储所有特征点的属性(FeaturePerId), 及其对应的帧属性
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size();
        // 特征点需要被两帧或两帧以上且都在滑动窗口队列内的图像同时观测到
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

// 检查视差, 用于确定这一帧图像是否为关键帧
// processImage() 函数读入一帧图像后, 首先会调用 f_manager.addFeatureCheckParallax() 检查该图像与前一帧图像之间的视差
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    LOGI("input feature: %d", (int)image.size());
    LOGI("num of feature: %d", getFeatureCount());
    
    // 当前帧的前两帧图像(2nd last and 3rd last)之间的视差总和
    double parallax_sum = 0;
    // 可以用于计算视差的特征点(能被当前帧和其前两帧共同观测到)个数
    int parallax_num = 0;
    // 对于读入的每一帧图像, last_track_num 现置0, 后满足条件后再累加
    // 用于说明这一帧图像有 last_track_num 个特征点在之前被其它的图像观测到过
    last_track_num = 0;

    // 遍历读入的这一帧图像的每一个特征点信息( 封装在image数组中, 格式: map< 特征点id, vector< pair< 相机id, 特征点信息Matrix > > > )
    for (auto &id_pts : image)
    {
        // 对每一个特征点创建一个帧属性(FeaturePerFrame)实例, 取出该特征点信息Matrix初始化该实例
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);
        // 取出该特征点id
        int feature_id = id_pts.first;
        // feature, list<FeaturePerId>类型数组, 存储所有特征点的属性(FeaturePerId), 及其对应的帧属性
        // 寻找 feature 数组中是否已存储该特征点, 并返回该特征点id
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });

        // 若 feature 数组中不存在该特征点, 就在 feature 数组中添加该特征点
        if (it == feature.end())
        {
            // start_frame 与第一次观测到该特征点图像的 frame_count 一致
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        // 若 feature 数组中已存在该特征点, 就在该特征点帧属性数组 feature_per_frame 中添加对应的帧属性 f_per_fra
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra); // feature_per_frame 该特征点帧属性数组, 观测到该特征点的所有图像对应的帧属性
            // 累加 last_track_num, 说明这一帧图像有 last_track_num 个特征点在之前被其它的图像观测到过
            last_track_num++;
        }
    }

    // 滑动窗口队列中图像少于两帧(是否计算视差意义不大), 或各个图像共同观测到的特征点太少(视差大, 导致视野不一样)
    // 返回 true, 直接置 marginalization_flag = MARGIN_OLD
    if (frame_count < 2 || last_track_num < 20)
        return true;

    // 遍历 feature 数组中的每一个特征点, 计算能被当前帧和其前两帧共同观测到的特征点视差
    for (auto &it_per_id : feature)
    {
        // start_frame 与第一帧观测到该特征点图像的 frame_count 一致
        // 能够观测到该特征点的图像序号满足: start_frame <= frame_count - 2 && endFrame() >= frame_count - 1
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            // 计算当前帧的前两帧图像(2nd last and 3rd last)之间的视差
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    // parallax_sum / parallax_num >= MIN_PARALLAX
    // 如果大于设定的阈值，则这一帧图像被判定为新的关键帧，而把滑动窗口队列中最老的一帧图像给marg掉
    // 如果小于设定的阈值，即前两帧图像之间没怎么运动，则把滑动窗口队列中倒数第二帧图像(2nd last)给丢掉，同时预积分和运动模型积分顺延
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        LOGI("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        LOGI("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    LOGI("debug show");
    for (auto &it : feature)
    {
        assert(it.feature_per_frame.size() != 0);
        assert(it.start_frame >= 0);
        assert(it.used_num >= 0);

        LOGI("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            LOGI("%d,", int(j.is_used));
            sum += j.is_used;
            LOGI("(%lf,%lf) ",j.point(0), j.point(1));
        }
        assert(it.used_num == sum);
    }
}

// 给定两帧图像 frame_count = frame_count_l, frame_count = frame_count_r
// 取出这两帧图像共同观测到的所有特征点的相机系归一化平面坐标 a, b, 并组合为 vector<pair<a, b>> 形式返回
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    // feature, list<FeaturePerId>类型数组, 存储所有特征点的属性(FeaturePerId), 及其对应的帧属性
    for (auto &it : feature)
    {
        // 如果给定的这两帧图像能观测到该特征点
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            // feature_per_frame 数组中索引关系参考 compensatedParallax2() 函数
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            // feature_per_frame 该特征点帧属性数组, 观测到该特征点的所有图像对应的帧属性
            // 取出该特征点投影到这两帧图像上的相机系归一化平面坐标
            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

// 给定feature 数组中被两帧或两帧以上且都在滑动窗口队列内的图像同时观测到的所有特征点的逆深度估计值向量 x
// 设置对应所有特征点在世界系下的深度估计值, optimization()函数之后的double2vector()函数调用
void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    // feature, list<FeaturePerId>类型数组, 存储所有特征点的属性(FeaturePerId), 及其对应的帧属性
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // 如果该特征点没有被两帧或两帧以上且都在滑动窗口队列内的图像同时观测到, 该特征点没有意义
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        // 设置该特征点在世界系下的深度估计值
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        
        // 0 haven't solve yet; 1 solve succ; 2 solve fail;
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

// 移除掉 feature 数组中 solve_flag = 2的特征点, 即深度值为负数的特征点
void FeatureManager::removeFailures()
{
    // feature, list<FeaturePerId>类型数组, 存储所有特征点的属性(FeaturePerId), 及其对应的帧属性
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        // 0 haven't solve yet; 1 solve succ; 2 solve fail;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

// 获取 feature 数组中被两帧或两帧以上且都在滑动窗口队列内的图像同时观测到的所有特征点的逆深度估计值
// 所有的逆深度估计值以向量 dep_vec 形式返回
VectorXd FeatureManager::getDepthVector()
{
    // getFeatureCount() 获取 feature 数组中被两帧或两帧以上且都在滑动窗口队列内的图像同时观测到的特征点总个数
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // 如果该特征点没有被两帧或两帧以上且都在滑动窗口队列内的图像同时观测到, 该特征点没有意义
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

// 视觉特征三角化, 被visualInitialAlign()函数和solveOdometry()函数调用
// 例如, 路标点 y 在若干帧图像 k = 1, ..., n 中被看到
// 已知投影关系: Pk = [Rk, tk], world系到camera系; xk = [uk, vk, 1]^T, 第k次观测该路标点在相机系归一化平面下的坐标
// 未知: 该特征点的深度值 depth(相机系特征点z坐标值), 该路标点在世界坐标系下的坐标 y = [xw, yw, zw, 1]^T
//          _  _                _   _
//         | uk |              | xw  |
// depth * | vk | = [Rk, tk] * | yw  |
//         |_ 1_|              | zw  |
//                             |_ 1 _|
// => depth * xk = Pk * y
// => depth = Pk(3)^T * y, Pk(3)^T为投影矩阵Pk第三行, 代入上面的式子
// => uk * Pk(3)^T * y = Pk(1)^T * y
//    vk * Pk(3)^T * y = Pk(2)^T * y
// => _                        _
//   |  uk * Pk(3)^T - Pk(1)^T  |
//   |_ vk * Pk(3)^T - Pk(2)^T _| * y = 0, 即超定方程 A*y = 0, A的维度为2n*4
// 对矩阵 A 进行SVD分解, 得到其最小奇异值对应的单位奇异向量(x',y',z',w')
// 单位奇异向量(x',y',z',w'), 不满足上面(xw, yw, zw,1)的齐次坐标形式, 根据齐次坐标定义, 只能取近似解
// 即 xw = x'/w', yw = y'/w', zw = z'/w', 1 = w'/w', 最终得到特征点在世界坐标系下的深度估计值 zw
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    // feature, list<FeaturePerId>类型数组, 存储所有特征点的属性(FeaturePerId), 及其对应的帧属性
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // 如果该特征点没有被两帧或两帧以上且都在滑动窗口队列内的图像同时观测到, 该特征点没有意义
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        // 如果该特征点深度估计值已存在, 则不用参与三角化处理    
        if (it_per_id.estimated_depth > 0)
            continue;
        
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        assert(NUM_OF_CAM == 1); // 要求为单目相机
        
        // 构建超定方程 A*y = 0, A的维度为2n*4
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        // R0 t0为第 imu_i 帧相机坐标系到世界坐标系的变换矩阵Rwc
        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        // feature_per_frame 该特征点帧属性数组, 观测到该特征点的所有图像对应的帧属性
        // 遍历观测到该特征点的每一帧图像
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            // R t为第 imu_j 帧相机坐标系到第 imu_i 帧相机坐标系的变换矩阵，P 为 imu_i 到 imu_j 的变换矩阵
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            // 取出路标点投影到这一帧图像上的相机系归一化平面坐标
            Eigen::Vector3d f = it_per_frame.point.normalized();
            // 构建超定方程 A*y = 0, A的维度为2n*4
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        
        // 构建超定方程 A*y = 0, 对矩阵 A 进行SVD分解, 得到其最小奇异值对应的单位奇异向量(x',y',z',w')，深度取近似值为 zw = z'/w'
        assert(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;
        
        // ???trick, 便于后面的BA优化
        // 若特征点在世界系下深度估计值太小(十分靠近相机), estimated_depth = INIT_DEPTH
        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}

void FeatureManager::removeOutlier()
{
    //ROS_BREAK();
    return;
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

// 计算当前帧的前两帧图像(2nd last and 3rd last)之间的视差
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{

    // 注意 feature_per_frame 数组中索引关系, 例如:
    // feature_per_frame数组索引:                   0 1 2 3
    // feature_per_frame数组元素(图像frame_count值): 7 8 9 10 (当前帧frame_count=10, start_frame=7, endFrame()=7+4-1=10)
    // (frame_count - 2)帧索引: 10 - 2 - 7 = 1
    // (frame_count - 1)帧索引: 10 - 1 - 7 = 2

    // check the second last frame is keyframe or not
    // parallax betwwen seconde last frame and third last frame
    // 取出第 (frame_count - 2)帧 与第 (frame_count - 1)帧图像的特征点信息, 并计算这两帧图像之间的视差计算
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point; // 特征点的相机系归一化平面坐标

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point; // 特征点的相机系归一化平面坐标
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    // 计算(相机系归一化平面下)特征点在这两帧中运动了多少个单位
    double du = u_i - u_j, dv = v_i - v_j;

    // 作者原意是想 p_i 由i帧重投影到j帧得到 p_i_comp, 然后计算(相机系归一化平面下)特征点在这两帧中运动了多少个单位
    // 但他注释掉了, 并置 p_i_comp = p_i, 故这一步与上一步重复, 实际上没有用
    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    // 实际计算欧氏距离 sqrt( dx*dx + dy*dy )
    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}