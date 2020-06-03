#include "system.h"

System* System::instance = nullptr;
ASensorEventQueue* System::accSensorEventQueue = nullptr;
ASensorEventQueue* System::gyrSensorEventQueue = nullptr;

System::System() {
    //LOGI("System Constructor");
    this->instance = this;
}
System::~System() {
    LOGI("System Destructor");
}

void System::Init() {
    isCapturing = false;
    bStart_backend = true;
    bool deviceCheck = SetParameters( DeviceType::Honor20 );
    if ( !deviceCheck ) {
        LOGE( "Device Not Supported!" );
    }
    trackerData[0].readIntrinsicParameter();
    estimator.setParameter();
    thd_BackEnd = std::thread( &System::ProcessBackEnd, this );
    ImuStartUpdate();
    isCapturing = true;
    LOGI( " VINS System Parameters were Initialized Successfuly!" );
}

// 对IMU和图像数据进行对齐对其并组合 pair< vector<IMU>, IMG >，返回 vector< pair< vector<IMU>, IMG > >
// 一帧IMG数据对应多帧IMU数据, 即[i, j]图像时刻中的所有图像和IMU数据, 取对齐好的第j帧图像和[i ,j]帧图像间的所有IMU数据进行组合
std::vector< std::pair< std::vector<ImuConstPtr>, ImgConstPtr > > System::GetMeasurements() {
    std::vector< std::pair< std::vector<ImuConstPtr>, ImgConstPtr > > measurements;

    while (true)
    {
        // 直到把imu_buf和feature_buf中的图像数据和IMU数据取完，才能够跳出此函数
        if ( imu_buf.empty() || feature_buf.empty() )
            return measurements;

        // imu_buf和feature_buf中数据的时间戳从front到back是递增的

        // imu_buf队尾处IMU数据时间戳大于feature_buf队首处图像数据，跳过此if语句
        if ( !(imu_buf.back()->header > feature_buf.front()->header + estimator.td) )
        {
            // imu_buf      : ... t2 t3 t4 t5 t6 t7 (no imu data)
            // feature_buf  :                                   ... t12 -- -- -- -- t17 ...
            // 这种情况应该继续等待获取更多的IMU测量数据, 只允许出现在处理的开始阶段
            LOGI("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        // imu_buf队首处IMU数据的时间戳小于feature_buf队首处图像数据，跳过此if语句
        if ( !(imu_buf.front()->header < feature_buf.front()->header + estimator.td) )
        {
            // imu_buf      :           ... t4 t5 t6 t7 t8 t9 ...
            // feature_buf  :  ... t1 -- -- -- -- t6 ...
            // 这种情况应该将旧的图像数据pop出队列,只允许出现在处理的开始阶段
            LOGI("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }

        // imu_buf      : ... t2 t3 t4 t5 t6 t7 ...
        // feature_buf  :       ... t4 -- -- -- -- t9 ...

        // 取出feature_buf队首处图像数据，并移出队列
        ImgConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        // 取出imu_buf中所有时间戳小于img_msg的IMU数据，并移出队列
        // 图像数据(img_msg)，对应多组在时间戳内的imu数据,然后塞入measurements
        std::vector<ImuConstPtr> IMUs;
        while ( imu_buf.front()->header < img_msg->header + estimator.td )
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }

        // !!!这里把下一帧IMU数据也放进去了,但没有pop出队列
        // !!!因此当前帧图像和下一帧图像会共用这一帧IMU数据(第一个时间戳大于当前帧图像时间戳的IMU数据)
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            LOGE( "NO IMU Between Two Image!" );
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

void System::ProcessBackEnd() {
    // 线程循环等待接收对齐好的IMU和图像数据
    while (bStart_backend)
    {
        //LOGI( "Thread thd_BackEnd, Function ProcessBackEnd..." );
        // getMeasurements() 对IMU和图像数据进行对齐对其并组合 vector< pair< vector<IMU>, IMG > >
        // 一帧IMG数据对应多帧IMU数据, 即[i, j]图像时刻中的所有图像和IMU数据, 取对齐好的第j帧图像和[i ,j]帧图像间的所有IMU数据进行组合
        std::vector< std::pair< std::vector<ImuConstPtr>, ImgConstPtr > > measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
        {
            return (measurements = GetMeasurements()).size() != 0;
        });
        lk.unlock();

        m_estimator.lock();
        // 遍历刚刚获取的IMU和图像数据
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            // 遍历刚刚获取的这一段IMU数据，并递推求解这一段时间末尾时刻(即图像帧时刻)对应的预积分量和运动模型积分得到的状态估计量
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header;
                double img_t = img_msg->header + estimator.td;
                // IMU数据时间戳早于图像数据
                // 一般仿真数据集中 getMeasurements() 返回的结果是IMU数据时间戳早于图像数据
                if (t <= img_t)
                {
                    if (current_time < 0)
                        current_time = t;
                    // dt 是两个IMU时刻之间的时间间隔
                    double dt = t - current_time;
                    assert(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x();
                    dy = imu_msg->linear_acceleration.y();
                    dz = imu_msg->linear_acceleration.z();
                    rx = imu_msg->angular_velocity.x();
                    ry = imu_msg->angular_velocity.y();
                    rz = imu_msg->angular_velocity.z();
                    // 对获取的IMU数据进行处理: 预积分更新, 运动模型更新
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //LOGI( "imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz );

                }
                // IMU数据时间戳晚于图像数据
                // 考虑鲁棒性，实际情况中还是会有两传感器(IMU和相机)的数据谁先收到和后收到的问题
                // 通常的一种情况是，两传感器时钟系统不同步且无规律可言，此时一帧图像的时间戳在两个IMU时刻之间
                // 那么就需要采用插值的方法，来对齐IMU和图像的位姿(四元数球面插值)和速度(单线性插值)
                else
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    assert(dt_1 >= 0);
                    assert(dt_2 >= 0);
                    assert(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x();
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y();
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z();
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x();
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y();
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z();
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //LOGI( "dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz );
                }
            }

            LOGI( "Processing vision data with stamp %f \n", img_msg->header );
            std::map< int, std::vector< std::pair< int, Eigen::Matrix<double, 7, 1> > > > image;
            // 遍历并获取这一帧图像数据中的每一个特征点信息，并存储到map数组 image 中
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->id_of_point[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                // 特征点去畸变校正后且在相机系归一化平面的坐标
                double x = img_msg->points[i].x();
                double y = img_msg->points[i].y();
                double z = img_msg->points[i].z();
                // 特征点像素系坐标
                double p_u = img_msg->u_of_point[i];
                double p_v = img_msg->v_of_point[i];
                // 特征点速度
                double velocity_x = img_msg->velocity_x_of_point[i];
                double velocity_y = img_msg->velocity_y_of_point[i];
                assert(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
                //LOGI( "dimg: point_3d: %f %f %f point_2d: %f %f velocity: %f %f \n",x, y, z, p_u, p_v, velocity_x, velocity_y );
            }

            // 对获取的这一帧图像数据进行处理: VIO初始化, 视觉特征三角化, BA非线性优化
            estimator.processImage(image, img_msg->header);
            if ( estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR ) {
                Vector3d p_wi;
                Quaterniond q_wi;
                q_wi = Quaterniond(estimator.Rs[WINDOW_SIZE]);
                p_wi = estimator.Ps[WINDOW_SIZE];
                double dStamp = estimator.Headers[WINDOW_SIZE];
                LOGI( "Timestamp: %f, Optimized location: %f, %f, %f", dStamp, p_wi.x(), p_wi.y(), p_wi.z() );
            }
        }
        m_estimator.unlock();

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        LOGI("Thread thd_BackEnd iteration done");
    }
}

void System::ImageStartUpdate(cv::Mat& image, double imgTimestamp, bool isScreenRotated) {

    if ( isCapturing ) {
        // 跳过第零帧
        if (!init_feature)
        {
            LOGI( "1 PubImageData skip the first detected feature, which doesn't contain optical flow speed" );
            init_feature = 1;
            return;
        }

        // 判断是否是第一帧图像
        if(first_image_flag)
        {
            LOGI( "Init the first image" );
            first_image_flag = false;
            first_image_time = imgTimestamp;
            last_image_time = imgTimestamp;
            return;
        }

        // detect unstable camera stream
        // 判断时间间隔，有问题(当前帧距上一帧发布时间久远，或当前帧时间戳早于上一帧)则 restart 重启
        if (imgTimestamp - last_image_time > 1.0 || imgTimestamp < last_image_time)
        {
            LOGI( "image discontinue! reset the feature tracker!" );
            first_image_flag = true;
            last_image_time = 0.0;
            pub_count = 1;
            return;
        }

        last_image_time = imgTimestamp;

        // frequency control
        // FREQ, frequence (Hz) of publish tracking result. At least 10Hz for good estimation.
        // If set FREQ == 0, the frequence will be same as raw image
        // 通过控制间隔时间 (img_msg->header.stamp.toSec() - first_image_time) 内的发布次数 pub_count
        // 来控制图像跟踪的发布频率不超过 FREQ (Hz)
        if ( round( 1.0 * pub_count / (imgTimestamp - first_image_time) ) <= FREQ) {
            PUB_THIS_FRAME = true;
            // reset the frequency control
            if (abs(1.0 * pub_count / (imgTimestamp - first_image_time) - FREQ) < 0.01 * FREQ) {
                first_image_time = imgTimestamp;
                pub_count = 0;
            }
        }
        else
            PUB_THIS_FRAME = false;

        // 单目 读取图像数据并进行特征跟踪处理
        //LOGI( "Start to process image via feature tracker!" );
        cv::cvtColor( image, image, cv::COLOR_BGRA2GRAY );
        trackerData[0].readImage(image, imgTimestamp);
        cv::cvtColor( image, image, cv::COLOR_GRAY2BGRA );

        for ( unsigned int i = 0;; i++ ) {
            bool completed = false;
            for (int j = 0; j < NUM_OF_CAM; j++)
                // 单目 更新当前帧图像特征点的全局ID
                if (j != 1 || !STEREO_TRACK)
                    completed |= trackerData[j].updateID(i);
            // 遍历完整个ids之后才会break，即updateID()返回false使completed = false 时才会break
            if (!completed)
                break;
        }

        // PUB_THIS_FRAME=1 封装该帧图像特征点信息并发布
        if (PUB_THIS_FRAME) {
            pub_count++;
            std::shared_ptr<IMG_MSG> feature_points( new IMG_MSG() );
            feature_points->header = imgTimestamp;
            vector<set<int>> hash_ids(NUM_OF_CAM);
            for (int i = 0; i < NUM_OF_CAM; i++) {
                auto &un_pts = trackerData[i].cur_un_pts;   // 特征点去畸变校正后且在相机系归一化平面的坐标
                auto &cur_pts = trackerData[i].cur_pts;     // 特征点像素坐标
                auto &ids = trackerData[i].ids;
                auto &pts_velocity = trackerData[i].pts_velocity;
                for (unsigned int j = 0; j < ids.size(); j++) {
                    if (trackerData[i].track_cnt[j] > 1) {
                        int p_id = ids[j];
                        hash_ids[i].insert(p_id);
                        double x = un_pts[j].x;
                        double y = un_pts[j].y;
                        double z = 1;
                        // 图像特征点信息封装
                        feature_points->points.push_back( Vector3d( x, y, z ) );
                        feature_points->id_of_point.push_back( p_id * NUM_OF_CAM + i );
                        feature_points->u_of_point.push_back( cur_pts[j].x );
                        feature_points->v_of_point.push_back( cur_pts[j].y );
                        // 图像特征点的移动速度（x,y方向单位时间内移动了多少个像素）
                        // 这个主要是用于IMU数据与图像数据的时间戳自动对齐（参考秦通18年IROS Paper）
                        feature_points->velocity_x_of_point.push_back( pts_velocity[j].x );
                        feature_points->velocity_y_of_point.push_back( pts_velocity[j].y );
                    }
                }
            }
            // skip the first image; since no optical speed on frist image
            if (!init_pub) {
                init_pub = 1;
            }
            else {
                m_buf.lock();
                // 前端特征点追踪结果 feature_points 最终存储到队列 System::feature_buf 中
                feature_buf.push( feature_points );
                m_buf.unlock();
                con.notify_one();
            }
        }

        if ( SHOW_TRACK ) {
            // 根据特征点被追踪的次数，显示其颜色，越红表示这个特征点看到的越久，一幅图像要是大部分特征点是蓝色，前端tracker效果很差了，估计要挂了
            for (unsigned int j = 0; j < trackerData[0].cur_pts.size(); j++)
            {
                double len = min( 1.0, 1.0 * trackerData[0].track_cnt[j] / WINDOW_SIZE );
                cv::circle( image, trackerData[0].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2 );
            }
        }

    }
    else {
        LOGI( "Not Caputuring!" );
    }
}

void System::ImuStartUpdate() {
    // Get a reference to the sensor manager
    ASensorManager *sensorManager = ASensorManager_getInstance();
    assert( sensorManager != NULL );

    ALooper* looper = ALooper_forThread();
    if( looper == NULL )
        looper = ALooper_prepare( ALOOPER_PREPARE_ALLOW_NON_CALLBACKS );
    assert(looper != NULL);

    // Creates a new sensor event queue and associate it with a looper
    accSensorEventQueue = ASensorManager_createEventQueue( sensorManager, looper, LOOPER_ID_USER, NULL, NULL );
    assert( accSensorEventQueue != NULL );
    // Returns the default sensor for the given type
    const ASensor *accelerometer = ASensorManager_getDefaultSensor( sensorManager, ASENSOR_TYPE_ACCELEROMETER );
    assert( accelerometer != NULL );

    auto status = ASensorEventQueue_enableSensor( accSensorEventQueue, accelerometer );
    assert(status >= 0);
    status = ASensorEventQueue_setEventRate( accSensorEventQueue, accelerometer, SENSOR_REFRESH_PERIOD_US );
    assert(status >= 0);

    gyrSensorEventQueue = ASensorManager_createEventQueue( sensorManager, looper, LOOPER_ID_USER, ProcessASensorEventsCallback, NULL);
    assert( gyrSensorEventQueue != NULL );
    const ASensor *gyroscope = ASensorManager_getDefaultSensor( sensorManager, ASENSOR_TYPE_GYROSCOPE );
    assert( gyroscope != NULL );

    status = ASensorEventQueue_enableSensor( gyrSensorEventQueue, gyroscope );
    assert(status >= 0);
    status = ASensorEventQueue_setEventRate( gyrSensorEventQueue, gyroscope, SENSOR_REFRESH_PERIOD_US );
    assert(status >= 0);

    LOGI("IMU EventQueues Initialized and Started!");
}

int System::ProcessASensorEventsCallback(int fd, int events, void* data) {
    static ASensorEvent accSensorEvent;
    ASensorEvent gyrSensorEvent;
    static double accTimestamp = -1.0;

    //LOGI( "Start to Update IMU Data!" );
    // Retrieve pending events in sensor event queue
    // Retrieve next available events from the queue to a specified event array
    while ( ASensorEventQueue_getEvents(gyrSensorEventQueue, &gyrSensorEvent, 1) > 0 ) {
        //LOGI( "Retrieve pending events in sensor event queue" );
        assert( gyrSensorEvent.type == ASENSOR_TYPE_GYROSCOPE );
        double gyrTimestamp = gyrSensorEvent.timestamp / 1000000000.0;
        assert( gyrTimestamp > 0  );

        IMU_MSG gyr_msg;
        gyr_msg.header = gyrTimestamp;
        gyr_msg.angular_velocity << gyrSensorEvent.uncalibrated_gyro.x_uncalib,
                                    gyrSensorEvent.uncalibrated_gyro.y_uncalib,
                                    gyrSensorEvent.uncalibrated_gyro.z_uncalib;
        //LOGI( "Filtering Gyroscope Data!" );
        if ( instance->gyr_buf.size() == 0 ) {
            instance->gyr_buf.push_back( gyr_msg );
            instance->gyr_buf.push_back( gyr_msg );
            continue;
        }
        else if ( gyr_msg.header <= instance->gyr_buf[1].header ) {
            continue;
        }
        else {
            instance->gyr_buf[0] = instance->gyr_buf[1];
            instance->gyr_buf[1] = gyr_msg;
        }

        // wait until gyroscope data is steady
        if ( instance->imu_prepare < 10 ) {
            LOGI( "Wait Until Gyroscope Data is Steady!" );
            instance->imu_prepare++;
            continue;
        }

        while ( accTimestamp < instance->gyr_buf[0].header  ) {
            ssize_t accEventNumbers;
            while ( (accEventNumbers = ASensorEventQueue_getEvents( accSensorEventQueue, &accSensorEvent, 1 )) == 0 ) {
                std::this_thread::sleep_for( std::chrono::milliseconds(1) );
            }
            assert( accEventNumbers == 1 );
            assert( accSensorEvent.type == ASENSOR_TYPE_ACCELEROMETER );

            accTimestamp = accSensorEvent.timestamp / 1000000000.0;

            std::shared_ptr<IMU_MSG> acc_msg(new IMU_MSG());
            acc_msg->header = accTimestamp;
            acc_msg->linear_acceleration << accSensorEvent.acceleration.x,
                                            accSensorEvent.acceleration.y,
                                            accSensorEvent.acceleration.z;
            instance->cur_acc = acc_msg;
        }

        if ( instance->gyr_buf[1].header < accTimestamp ) {
            LOGE("Have to Wait for Fitting Gyroscope Event!");
            continue;
        }

        // 插值
        //LOGI( "Interpolation Between Gyroscope and Accelerometer" );
        std::shared_ptr<IMU_MSG> imu_msg(new IMU_MSG());
        if ( instance->cur_acc->header >= instance->gyr_buf[0].header && instance->cur_acc->header < instance->gyr_buf[1].header ) {

            imu_msg->header = instance->cur_acc->header;
            imu_msg->linear_acceleration = instance->cur_acc->linear_acceleration;
            imu_msg->angular_velocity = instance->gyr_buf[0].angular_velocity +
                                        ( instance->gyr_buf[1].angular_velocity - instance->gyr_buf[0].angular_velocity ) *
                                        ( instance->cur_acc->header - instance->gyr_buf[0].header ) / ( instance->gyr_buf[1].header - instance->gyr_buf[0].header );

        }
        else {
            LOGE( "IMU Data Error, Current Accelerometer Data Error!" );
            continue;
        }

        instance->m_buf.lock();
        instance->imu_buf.push( imu_msg );
        instance->m_buf.unlock();
        instance->con.notify_one();
    }
    //should return 1 to continue receiving callbacks, or 0 to unregister
    return 1;
}

void System::ImuStopUpdate() {
    ASensorManager *sensorManager = ASensorManager_getInstance();
    if ( sensorManager != NULL ) {
        const ASensor *accelerometer = ASensorManager_getDefaultSensor(sensorManager, ASENSOR_TYPE_ACCELEROMETER);
        const ASensor *gyroscope = ASensorManager_getDefaultSensor(sensorManager, ASENSOR_TYPE_GYROSCOPE);
        ASensorEventQueue_disableSensor(accSensorEventQueue, accelerometer);
        ASensorEventQueue_disableSensor(gyrSensorEventQueue, gyroscope);
    }
}

