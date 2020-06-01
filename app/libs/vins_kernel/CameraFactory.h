#ifndef CAMERAFACTORY_H
#define CAMERAFACTORY_H

#include <opencv2/core/core.hpp>

#include "Camera.h"

namespace camodocal
{

class CameraFactory
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CameraFactory();

    static std::shared_ptr<CameraFactory> instance(void);

    CameraPtr generateCamera( Camera::ModelType modelType, const std::string& cameraName,
                              cv::Size imageSize, cv::Mat K, cv::Mat D ) const;

private:
    static std::shared_ptr<CameraFactory> m_instance;
};

}

#endif
