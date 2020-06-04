#include "draw_result.h"

DrawResult::DrawResult( float _yall, float _pitch, float _roll, float _T_x, float _T_y, float _T_z )
    : yaw(_yall), pitch(_pitch), roll(_roll), Tx(_T_x), Ty(_T_y), Tz(_T_z)
{
    theta = 75;
    phy = 89;
    radius = 5.0;
    origin_w.setZero();
    Fx = 400;
    Fy = 400;
    X0 = WIDTH/2;
    Y0 = HEIGHT/2;
}

DrawResult::~DrawResult() {}

bool DrawResult::checkBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < WIDTH - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < HEIGHT - BORDER_SIZE;
}

std::vector<Eigen::Vector3f> DrawResult::CalculateCameraPose( Eigen::Vector3f camera_center,
        Eigen::Matrix3f Rc, float length )
{
    vector<Vector3f> result;
    vector<Vector3f> origin;
    origin.push_back(Vector3f(length/2.0,length/2.0,-length*0.7));
    origin.push_back(Vector3f(-length/2.0,length/2.0,-length*0.7));
    origin.push_back(Vector3f(-length/2.0,-length/2.0,-length*0.7));
    origin.push_back(Vector3f(length/2.0,-length/2.0,-length*0.7));
    origin.push_back(Vector3f(0,0,0));
    result.clear();

    Eigen::Matrix3f RIC;
    RIC = Utility::ypr2R(Vector3d(RIC_y,RIC_p,RIC_r)).cast<float>();

    for (auto it : origin)
    {
        Vector3f tmp;
        tmp = Rc*it + camera_center;
        /*
         Eigen::Vector3f Pc;
         cv::Point2f pts;
         Pc = R_c_w * RIC.transpose() * tmp + T_c_w;
         pts.x = Fx * Pc.x() / Pc.z()+ PX;
         pts.y = Fy * Pc.y() / Pc.z()+ PY;
         */
        result.push_back(tmp);
    }
    return result;
}

cv::Point2f DrawResult::World2VirturCam( Eigen::Vector3f xyz, float &depth ) {
    Vector3f camInWorld_T;
    if( phy > 89 )
        phy = 89;
    else if( phy < -89 )
        phy = -89;
    if( theta > 89 )
        theta = 89;
    else if( theta < -89 )
        theta = -89;

    camInWorld_T.z() = radius * sin(theta * C_PI/ 180.0);
    camInWorld_T.x() = -radius * cos(theta * C_PI/ 180.0) * sin(phy* C_PI/ 180.0);
    camInWorld_T.y() = -radius * cos(theta* C_PI/ 180.0) * cos(phy* C_PI/ 180.0);
    Matrix3f camInWorld_R;
    // make sure camera optical axis is towards to world origin
    // camInWorld_T = -camInWorld_R * (0, 0, 1)^T
    // camInWorld_R = Utility::ypr2R(Vector3f(0, 0, -theta)) * Utility::ypr2R(Vector3f(0, phy, 0)) * Utility::ypr2R(Vector3f(0, 0, -90));
    Vector3f Zwc = -camInWorld_T/camInWorld_T.lpNorm<2>();
    Vector3f Xwc;
    Xwc << 1.0, -camInWorld_T.x()/camInWorld_T.y(), 0;
    Xwc = Xwc/Xwc.lpNorm<2>();
    Vector3f Ywc = Zwc.cross(Xwc);
    Ywc = Ywc/Ywc.lpNorm<2>();

    camInWorld_R << Xwc.x(),Ywc.x(),Zwc.x(),
                    Xwc.y(),Ywc.y(),Zwc.y(),
                    Xwc.z(),Ywc.z(),Zwc.z();

    Vector3f Pc = camInWorld_R.transpose() * (xyz - origin_w - camInWorld_T);

    cv::Point2f pts;
    pts.x = Fx * Pc.x() / Pc.z()+ Y0;
    pts.y = Fy * Pc.y() / Pc.z()+ X0;
    depth = Pc.z();
    return pts;
}

// Reproject Pw to image plane of a virtual camera(手动投影，仅用 opencv 绘制坐标系与实时轨迹)
// result, 用于显示轨迹的image plane
// point_cloud, 优化之后滑窗中所有待显示的路标点3D坐标
// R_window, t_window,优化之后滑窗中所有待显示的相机位姿
void DrawResult::Reprojection( cv::Mat& result, std::vector <Vector3f>& point_cloud,
        const Eigen::Matrix3f* R_window, const Eigen::Vector3f* t_window )
{
    float depth_marker;
    cv::Mat aa( HEIGHT, WIDTH, CV_8UC3, Scalar(242,242,242) );
    result = aa;

    Eigen::Matrix3f RIC;
    RIC = Utility::ypr2R(Vector3d(RIC_y,RIC_p,RIC_r)).cast<float>();
    Eigen::Matrix3f R_v_c = Utility::ypr2R(Eigen::Vector3f{yaw, pitch, roll});
    Eigen::Vector3f T_v_c;
    T_v_c << Tx, Ty, Tz;

    cv::Point2f pts_pre;
    cv::Point2f pts;
    // 绘制轨迹
    for (int i=0; i<pose.size(); i++)
    {
        Eigen::Vector3f Pc;
        pts = World2VirturCam(pose[i], depth_marker);
        if(i == 0)
        {
            pts_pre = pts;
            continue;
        }
        cv::line( result, pts_pre, pts, cv::Scalar(255, 0, 0), 2, 8, 0 );
        pts_pre = pts;
    }

    // 绘制坐标系箭头
    {
        Vector3f p1, p2;
        cv::Point2f pt1, pt2;
        float length = 2.4 * 400 * radius / (Fx * 5.0) ;
        float scale_factor;
        p1 << -length, 0, 0;
        p2 << length, 0, 0;
        pt1 = World2VirturCam(p1, depth_marker);
        pt2 = World2VirturCam(p2, depth_marker);

        cv::arrowedLine(result, pt1, pt2, cv::Scalar(100,100,100),1, 8, 0, 0.02);
        cv::putText(result, "X", pt2, 0, 0.5, cv::Scalar(100,100,100));

        p1 << 0, -length, 0;
        p2 << 0, length, 0;
        pt1 = World2VirturCam(p1, depth_marker);
        pt2 = World2VirturCam(p2, depth_marker);

        cv::arrowedLine(result, pt1, pt2, cv::Scalar(100,100,100),1 , 8, 0, 0.02);
        cv::putText(result, "Y", pt2, 0, 0.5, cv::Scalar(100,100,100));

        p1 << 0, 0, -length;
        p2 << 0, 0, length;
        pt1 = World2VirturCam(p1, depth_marker);
        pt2 = World2VirturCam(p2, depth_marker);

        cv::arrowedLine(result, pt1, pt2, cv::Scalar(100,100,100), 1, 8, 0, 0.02);
        cv::putText(result, "Z", pt2, 0, 0.5, cv::Scalar(100,100,100));

        // 绘制栅格
        {
            float dis = 1.0;
            int line_num = 9;
            vector<std::pair<Vector3f, Vector3f>> grid_space;
            vector<pair<Point2f, Point2f>> grid_plane;
            Vector3f origin_grid;
            origin_grid << -dis*(line_num/2), -dis*(line_num/2), 0;
            for(int i=0; i < line_num; i++)
            {
                std::pair<Vector3f, Vector3f> tmp_Pts;
                tmp_Pts.first = origin_grid + Vector3f(dis * i, 0, 0);
                tmp_Pts.second = origin_grid + Vector3f(dis * i, dis*(line_num - 1), 0);
                grid_space.push_back(tmp_Pts);

                tmp_Pts.first = origin_grid + Vector3f(0, dis * i, 0);
                tmp_Pts.second = origin_grid + Vector3f(dis*(line_num - 1), dis * i, 0);
                grid_space.push_back(tmp_Pts);
            }
            for(auto it : grid_space)
            {
                cv::Point2f pts;
                pts = World2VirturCam(it.first, depth_marker);

                cv::Point2f pts2;
                pts2 = World2VirturCam(it.second, depth_marker);
                cv::line(result, pts, pts2, cv::Scalar(180,180,180), 1, 8, 0);
            }
        }
    }

    // 绘制相机示意框
    for(int i = 0; i < WINDOW_SIZE; i++)
    {
        float length;
        if(i == WINDOW_SIZE - 1)
            length = 0.3;
        else
            length = 0.1;
        vector<Vector3f> camera_coner_w = CalculateCameraPose( t_window[i], R_window[i], length );
        vector<cv::Point2f> camera_coner;
        for(auto it : camera_coner_w)
        {
            camera_coner.push_back(World2VirturCam(it, depth_marker));
        }

        cv::Scalar camera_color = cv::Scalar(0,0,255);
        cv::line(result, camera_coner[0], camera_coner[1], camera_color, 1, 8, 0);  //RGB
        cv::line(result, camera_coner[1], camera_coner[2], camera_color, 1, 8, 0);
        cv::line(result, camera_coner[2], camera_coner[3], camera_color, 1, 8, 0);
        cv::line(result, camera_coner[3], camera_coner[0], camera_color, 1, 8, 0);

        cv::line(result, camera_coner[4], camera_coner[0], camera_color, 1, 8, 0);
        cv::line(result, camera_coner[4], camera_coner[1], camera_color, 1, 8, 0);
        cv::line(result, camera_coner[4], camera_coner[2], camera_color, 1, 8, 0);
        cv::line(result, camera_coner[4], camera_coner[3], camera_color, 1, 8, 0);
    }

    // 绘制3D点云
    for (int i=0; i<point_cloud.size(); i++)
    {
        Eigen::Vector3f Pc;
        Pc = RIC.transpose()*point_cloud[i];
        Eigen::Vector3f Pv;
        pts = World2VirturCam(point_cloud[i], depth_marker);
        if(checkBorder(pts))
        {
            cv::circle(result, pts, 0, cv::Scalar(0,255,0), 3);
        }
    }
}
