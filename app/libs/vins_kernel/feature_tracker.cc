#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

// 给定特征点的2D像素系坐标，判断该特征点在图像边界内部，还是在外部
bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

// 依次取出 v 中status为1的元素，按序返存回 v 中，并 resize 数组 v 的大小
// 即根据 cv::calcOpticalFlowPyrLK()或cv::goodFeaturesToTrack()返回的 status 剔除掉值为0外点
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

// 依次取出 v 中status为1的元素，按序返存回 v 中，并 resize 数组 v 的大小
// 即根据 cv::calcOpticalFlowPyrLK()或cv::goodFeaturesToTrack()返回的 status 剔除掉值为0外点
void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}

// 对跟踪到的特征点按照被跟踪的次数从大到小进行排序
// 取出密集跟踪的点使特征点均匀分布
void FeatureTracker::setMask()
{
    // 鱼眼相机采用特定的mask
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    // 对跟踪到的特征点 forw_pts 按照被跟踪的次数 track_cnt 从大到小进行排序
    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    // 对密集跟踪的特征点，半径为MIN_DIST圆形区域内不再取其它的跟踪点，使特征点均匀分布
    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            // cv::circle 绘制圆圈，thickness=-1意味着填充整个圆圈
            // 即mask中，该特征点处为圆心，半径为MIN_DIST的圆形区域内填充值为0，使跟踪的特征点不集中在一个区域上
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

// 向 forw_pts 添加新的特征点
// 这些新特征点的id皆初始化为-1，track_cnt皆初始化为1
void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

// 核心函数，读取图像并进行特征跟踪处理
// LK光流跟踪的过程:
// 1. 第一帧读入的图像直接跳转到 cv::goodFeaturesToTrack() 中检测角点
// 2. 接下来读入的图像调用 cv::calcOpticalFlowPyrLK() 进行光流跟踪, 并剔除跟踪失败的点
// 3. 利用对极约束(基础矩阵F)剔除外点, 并剔除分布密集的点, 再调用 cv::goodFeaturesToTrack() 补充新的角点用于下一次跟踪
// 4. 将本次跟踪结果(跟踪成功的特征点+补充的新特征点)进行去畸变校正, 并转换到相机系归一化平面, 最后输出返回
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    cur_time = _cur_time;

    // EQUALIZE = 1, cv::createCLAHE 自适应直方图均衡化处理
    if (EQUALIZE)
    {
        // 直方图柱子高度大于计算后clipLimit的部分被裁剪掉，然后将其平均分配给整张直方图从而提升整个图像
        // clipLimit = 3.0, 将图像分为8*8块
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(_img, img);
    }
    else
        img = _img;

    // 命名规则:
    // readImage()函数读入的当前帧数据以 forw_xxx 形式命名(跟踪之前为输入的当前帧, 跟踪之后赋值给cur_xxx输出)
    // readImage()函数输出的当前帧数据以 cur_xxx 形式命名(跟踪之前为前一帧, 跟踪之后为输出的当前帧)
    // readImage()函数末尾会将当前帧数据以 cur_xxx = forw_xxx; 的形式输出
    // readImage()函数末尾会将前一帧数据以 prev_xxx = cur_xxx; 的形式暂存

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        vector<uchar> status;
        vector<float> err;
        // 使用具有金字塔的迭代Lucas-Kanade方法计算稀疏特征集的光流
        // cur_img 第一帧8 bit图像或金字塔
        // forw_img 与cur_img相同大小和类型的第二帧图像或金字塔
        // cur_pts 计算光流所需要的输入2D点矢量, 点坐标必须是单精度浮点数
        // forw_pts 输出2D点矢量
        // status 输出状态向量, 如果该特征点光流跟踪上了，则向量对应位置处元素设置为1，否则设置为0
        // err 输出误差向量
        // cv::Size(21, 21) 每帧图像或金字塔搜索窗口winSize的大小, 如果置1, 金字塔2层, 等等以此类推
        // maxLevel=3 基于0的最大金字塔等级数, 如果设置为0，则不使用金字塔（单级）
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_pts.size()); i++)
            // 不在图像(含边界)内部的特征点，对应 status 置 0
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        
        // 根据 cv::calcOpticalFlowPyrLK() 返回的 status，剔除跟踪失败和图像边界外部的点
        // 例如，依次取出 prev_pts 中status为1的元素，按序返存回 prev_pts 中，并 resize 数组 prev_pts 的大小

        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);          // ids 记录特征点id
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);    // track_cnt 记录被跟踪的次数
    }

    for (auto &n : track_cnt)
        n++;

    // PUB_THIS_FRAME=1 表示该帧图像特征点信息最后会发布给后端进行优化
    if (PUB_THIS_FRAME)
    {
        // 通过对极约束中的基础矩阵 F 剔除外点
        rejectWithF();
        LOGI("set mask begins");
        // 对跟踪到的特征点按照被跟踪的次数从大到小进行排序
        // 取出密集跟踪的点使特征点均匀分布
        setMask();

        LOGI("detect feature begins");
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            // 寻找新的特征点，使得跟踪的特征点数达到 MAX_CNT
            // cv::goodFeaturesToTrack 可以计算Harris角点和shi-tomasi角点，但默认情况下计算的是shi-tomasi角点
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();

        LOGI("add feature begins");
        // 向 forw_pts 添加新的特征点
        // 这些新特征点的id皆初始化为-1，track_cnt皆初始化为1
        addPoints();
    }
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    // 对特征点进行去畸变校正, 对特征点相机系坐标进行归一化处理, 计算每一个特征点的速度
    undistortedPoints();
    prev_time = cur_time;
}

// 通过对极约束中的基础矩阵 F 剔除外点
void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        LOGI("FM ransac begins");
        // 对 cur_pts 和 forw_pts 中的特征点进行去畸变校正
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            
            // liftProjective 对针孔相机而言，将特征点像素系2D坐标，转换为相机系归一化平面坐标，去畸变校正后返回

            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        // 采用 RANSAC 迭代求解得到图像 cur 与 forw 之间的基础矩阵 F(并不需要)
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        // 根据 cv::findFundamentalMat()函数返回的 status 剔除掉外点，即利用对极约束剔除外点
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        LOGI("FM ransac: %d -> %u: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
    }
}

// 更新特征点的全局id
// 每一帧图像cv::goodFeaturesToTrack()检测的新特征点id初始值为-1
// 从readImage()函数读入的第一帧图像开始, 所有特征点id随n_id由0开始累加
bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        // ids[i] = -1 只有两种情况:
        // 这帧图像为初始第一帧图像，所有特征点为新检测的角点
        // 这帧图像所补充的新特征点
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

// 创建一个相机模型camodocal::CameraPtr对象, 读取calib_file中的相机相关参数来配置该对象属性
void FeatureTracker::readIntrinsicParameter()
{
    m_camera = CameraFactory::instance()->generateCamera( CAMERA_TYPE, CAMERA_NAME, CAMERA_SIZE, CAMERA_K, CAMERA_D );
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            // liftProjective 对针孔相机而言，将特征点像素系2D坐标，转换为相机系归一化平面坐标，去畸变校正后返回
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //LOGI("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //LOGI("%f %f\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //LOGI("%f %f\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //LOGE("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

// 对特征点进行去畸变校正, 对特征点相机系坐标进行归一化处理, 计算每一个特征点的速度
// 特征点速度是利用特征点相机系归一化坐标进行计算得到
void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    
    // 对特征点进行去畸变校正
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());

    // 对特征点相机系坐标进行归一化处理
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        // liftProjective 对针孔相机而言，将特征点像素系2D坐标，转换为相机系归一化平面坐标，去畸变校正后返回
        m_camera->liftProjective(a, b);
        // 非针孔相机则需要对相机系坐标进行归一化处理
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //LOGI("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }

    // caculate points velocity
    // 计算每一个特征点的速度
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            // ids[i] = -1 只有两种情况:
            // 这帧图像为初始第一帧图像，所有特征点为新检测的角点
            // 这帧图像所补充的新特征点
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    // 特征点速度是利用特征点相机系归一化坐标进行计算得到
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
