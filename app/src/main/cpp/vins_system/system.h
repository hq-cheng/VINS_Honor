#ifndef VINS_HONOR_SYSTEM_H
#define VINS_HONOR_SYSTEM_H

#include <queue>
#include <vector>
#include <set>
#include <map>
#include <jni.h>
#include <android/log.h>
#include <android/looper.h>
#include <android/sensor.h>
#include <condition_variable>
#include <thread>
#include <chrono>
#include "Parameters.h"
#include "feature_tracker.h"
#include <opencv2/core/core.hpp>

#define LOG_TAG "system.cpp"
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

struct IMU_MSG {
    double header;
    Eigen::Vector3d linear_acceleration;
    Eigen::Vector3d angular_velocity;
};
typedef std::shared_ptr<IMU_MSG const> ImuConstPtr;

struct IMG_MSG {
    double header;
    std::vector<Vector3d> points;
    std::vector<int> id_of_point;
    std::vector<float> u_of_point;
    std::vector<float> v_of_point;
    std::vector<float> velocity_x_of_point;
    std::vector<float> velocity_y_of_point;
};
typedef std::shared_ptr <IMG_MSG const> ImgConstPtr;

class System {
public:
    System();
    ~System();

    void Init();
    void ImuStartUpdate();
    void ImuStopUpdate();
    void ImageStartUpdate(cv::Mat& image, double imgTimestamp, bool isScreenRotated);
    std::vector< std::pair< std::vector<ImuConstPtr>, ImgConstPtr > > GetMeasurements();
    void ProcessBackEnd();

private:
    static System* instance;

    static ASensorEventQueue* accSensorEventQueue;
    static ASensorEventQueue* gyrSensorEventQueue;
    static int ProcessASensorEventsCallback(int fd, int events, void* data);

private:
    std::condition_variable con;
    std::mutex m_buf;
    std::mutex m_estimator;

    std::queue<ImgConstPtr> feature_buf;
    std::queue<ImuConstPtr> imu_buf;

    // for feature_buf
    bool isCapturing;
    bool init_pub = 0;
    bool init_feature = 0;
    bool first_image_flag = true;
    double first_image_time;
    double last_image_time = 0.0;
    int pub_count = 1;

    // for imu_buf
    const int LOOPER_ID_USER = 3;
    const int SENSOR_REFRESH_RATE_HZ = 100;
    const int32_t SENSOR_REFRESH_PERIOD_US = int32_t(1000000 / SENSOR_REFRESH_RATE_HZ);
    std::shared_ptr<IMU_MSG> cur_acc = std::shared_ptr<IMU_MSG>(new IMU_MSG());
    std::vector<IMU_MSG> gyr_buf;
    int imu_prepare = 0;

    // for back-end
    std::thread thd_BackEnd;
    bool bStart_backend;
    FeatureTracker trackerData[1];

    int sum_of_wait = 0;
    double current_time = -1;


};

#endif //VINS_HONOR_SYSTEM_H
