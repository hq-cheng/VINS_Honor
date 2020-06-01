#include "CameraFactory.h"
#include "PinholeCamera.h"
#include <string>

namespace camodocal
{

std::shared_ptr<CameraFactory> CameraFactory::m_instance;

CameraFactory::CameraFactory()
{

}

std::shared_ptr<CameraFactory>
CameraFactory::instance(void)
{
    if (m_instance.get() == 0)
    {
        m_instance.reset(new CameraFactory);
    }

    return m_instance;
}

CameraPtr
CameraFactory::generateCamera( Camera::ModelType modelType, const std::string& cameraName,
                               cv::Size imageSize, cv::Mat K, cv::Mat D ) const
{
    switch (modelType)
    {
    case Camera::KANNALA_BRANDT:
    case Camera::PINHOLE:
    {
        PinholeCameraPtr camera(new PinholeCamera);

        PinholeCamera::Parameters params = camera->getParameters();
        params.cameraName() = cameraName;
        params.imageWidth() = imageSize.width;
        params.imageHeight() = imageSize.height;
        params.fx() = K.at<double>(0, 0);
        params.fy() = K.at<double>(1, 1);
        params.cx() = K.at<double>(0, 2);
        params.cy() = K.at<double>(1, 2);
        params.k1() = D.at<double>(0, 0);
        params.k2() = D.at<double>(0, 1);
        params.p1() = D.at<double>(0, 2);
        params.p2() = D.at<double>(0, 3);
        camera->setParameters(params);
        return camera;
    }
    case Camera::SCARAMUZZA:
    case Camera::MEI:
    default:
        return CameraPtr();
    }
}

}

