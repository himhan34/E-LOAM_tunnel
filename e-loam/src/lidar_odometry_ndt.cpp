#include <queue>
#include <mutex>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h> 

#include<time.h>
#include <pclomp/ndt_omp.h>
#include "e_loam/lidarFactor.hpp"

// NDT 결과를 저장할 포인트 클라우드
pcl::PointCloud<pcl::PointXYZI>::Ptr NDTPoints(new pcl::PointCloud<pcl::PointXYZI>());
// pcl::PointCloud<pcl::PointXYZI>::Ptr NDTPointsLess(new pcl::PointCloud<pcl::PointXYZI>());

// 포인트 클라우드 메시지 버퍼를 관리할 큐
std::queue<sensor_msgs::PointCloud2ConstPtr> NDTBuf;
// std::queue<sensor_msgs::PointCloud2ConstPtr> NDTLessBuf;

// 이전 레이저 클라우드 포인트
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudNDTLast(new pcl::PointCloud<pcl::PointXYZI>());

// Normal Distributions Transform(NDT) 알고리즘 객체
pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>::Ptr ndt(new pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>());

// 현재 프레임에서 월드 프레임으로의 회전을 나타내는 쿼터니언
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
// 현재 프레임에서 월드 프레임으로의 위치 변환 벡터
Eigen::Vector3d t_w_curr(0, 0, 0);

// 이전 프레임에서 현재 프레임으로의 NDT 초기 추정 회전을 나타내는 쿼터니언
Eigen::Quaterniond q_last_curr_ndt(1, 0, 0, 0);
// 이전 프레임에서 현재 프레임으로의 NDT 초기 추정 위치 변환 벡터
Eigen::Vector3d t_last_curr_ndt(0, 0, 0);

// 초기 추정값을 포함하는 변환 행렬
Eigen::Matrix4f init_guess;

// NDT 연산 시간
double timeNDT = 0;
// 다른 NDT 연산 변수 또는 결과를 저장하는 변수
// double timeNDTLess = 0;
double score = 0;

// 레이저 오도메트리 데이터를 퍼블리시하는 객체
ros::Publisher pubLaserOdometry;
// 레이저 경로 정보를 퍼블리시하는 객체
ros::Publisher pubLaserPath;
// NDT 클라우드 정보를 퍼블리시하는 객체
ros::Publisher pubNDTCloud;

// 레이저 경로 정보를 저장하는 ROS 메시지 객체
nav_msgs::Path laserPath;

// 시스템 초기화 여부를 나타내는 변수
bool systemInited = false;

// 데이터 버퍼에 대한 뮤텍스 (스레드 동기화) 객체
std::mutex mBuf;


// NDT 클라우드 데이터를 받아서 데이터 버퍼에 저장하는 콜백 함수
void subNDTCallback(const sensor_msgs::PointCloud2ConstPtr& point_msgs)
{
    // 데이터 버퍼에 접근하기 위한 뮤텍스를 잠금
    mBuf.lock();
    // NDT 클라우드 메시지를 데이터 버퍼에 추가
    NDTBuf.push(point_msgs);
    // 뮤텍스 잠금 해제
    mBuf.unlock();
}

// 다른 콜백 함수의 주석 처리된 부분
// void subNDTLessCallback(const sensor_msgs::PointCloud2ConstPtr& point_msgs)
// {
//     mBuf.lock();
//     NDTLessBuf.push(point_msgs);
//     mBuf.unlock();
// }


int main(int argc, char** argv) 
{
    // ROS 노드 초기화
    ros::init(argc, argv, "test_odometry_ndt_node");
    ros::NodeHandle node = ros::NodeHandle();

    // NDT 객체 설정
    // ndt->setTransformationEpsilon(1e-6); // 변환 엡실론 설정 (선택적)
    ndt->setResolution(4.0); // NDT 해상도 설정
    ndt->setStepSize(0.05); // NDT 스텝 사이즈 설정
    ndt->setMaximumIterations(35); // 최대 반복 횟수 설정
    ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7); // 이웃 검색 방법 설정
    ndt->setNumThreads(4); // 병렬 처리 스레드 수 설정


    ros::Subscriber pointcloudNDTSub = node.subscribe("/laser_cloud_NDT", 5, &subNDTCallback); // NDT 포인트 클라우드 구독
    // ros::Subscriber pointcloudNDTLessSub = node.subscribe("/NDTPointLess", 5, &subNDTLessCallback); // 다른 NDT 포인트 클라우드 구독 (선택적)
    
    pubNDTCloud = node.advertise<sensor_msgs::PointCloud2>("ndt_cloud1", 5); // NDT 결과 포인트 클라우드 발행
    pubLaserOdometry = node.advertise<nav_msgs::Odometry>("/laser_odom_ndt", 5); // NDT 결과 오도메트리 발행
    
    pubLaserPath = node.advertise<nav_msgs::Path>("/laser_odom__ndt_path", 5); // NDT 결과 경로 발행
    
    ros::Rate rate(100); // 노드의 주기 설정


    while(ros::ok())
    {
        ros::spinOnce(); // ROS 이벤트 처리
    
        if(!NDTBuf.empty()) // NDT 버퍼에 데이터가 있는지 확인
        {
            timeNDT = NDTBuf.front()->header.stamp.toSec(); // NDT 포인트 클라우드 메시지의 타임스탬프 가져오기
    
            mBuf.lock(); // 버퍼 락
            NDTPoints->clear(); // NDT 포인트 클라우드 초기화
            pcl::fromROSMsg(*NDTBuf.front(), *NDTPoints); // ROS 메시지를 PCL 포인트 클라우드로 변환
            NDTBuf.pop(); // 버퍼에서 메시지 제거
    
            // NDTPointsLess->clear();
            // pcl::fromROSMsg(*NDTLessBuf.front(), *NDTPointsLess);
            // NDTLessBuf.pop();
            mBuf.unlock(); // 버퍼 언락




            
            auto t1 = ros::WallTime::now();
            
            if (!systemInited) // 시스템이 초기화되지 않았을 때
            {
                systemInited = true;
                std::cout << "Initialization finished" << std::endl;
            }
            else
            {
                ndt->setInputCloud(NDTPoints); // 현재 포인트 클라우드를 NDT 알고리즘의 입력으로 설정
                ndt->setInputTarget(laserCloudNDTLast); // 이전 포인트 클라우드를 NDT 알고리즘의 타겟으로 설정
                pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>());
                ndt->align(*aligned); // NDT 알고리즘을 사용하여 포인트 클라우드 정합
            
                score = ndt->getFitnessScore(); // 정합의 점수를 가져옴
                Eigen::Quaternionf tmp_q(ndt->getFinalTransformation().topLeftCorner<3, 3>());
                Eigen::Vector3f tmp_t(ndt->getFinalTransformation().topRightCorner<3, 1>());
                q_last_curr_ndt = tmp_q.cast<double>();
                t_last_curr_ndt = tmp_t.cast<double>();
               
                t_w_curr = t_w_curr + q_w_curr * t_last_curr_ndt; // 월드 좌표계에서 현재 위치 업데이트
                q_w_curr = q_w_curr * q_last_curr_ndt; // 월드 좌표계에서 현재 회전 업데이트
            
                init_guess = ndt->getFinalTransformation(); // 초기 추정 업데이트
            }

            auto t2 = ros::WallTime::now();
            
            // 라이다 오도메트리 시간 측정
            std::cout << "lidar odometry_ndt: " << (t2 - t1).toSec() * 1000 << "  ms" << std::endl;
            
            // 현재 포인트 클라우드와 이전 포인트 클라우드를 교환
            pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudTemp = NDTPoints;
            NDTPoints = laserCloudNDTLast;
            laserCloudNDTLast = laserCloudTemp;
            
            // 오도메트리 정보를 게시
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "/odom";
            laserOdometry.child_frame_id = "/velodyne";
            laserOdometry.header.stamp = ros::Time::now();
            laserOdometry.pose.pose.orientation.x = q_last_curr_ndt.x(); // 현재 위치의 회전 정보 (x)
            laserOdometry.pose.pose.orientation.y = q_last_curr_ndt.y(); // 현재 위치의 회전 정보 (y)
            laserOdometry.pose.pose.orientation.z = q_last_curr_ndt.z(); // 현재 위치의 회전 정보 (z)
            laserOdometry.pose.pose.orientation.w = q_last_curr_ndt.w(); // 현재 위치의 회전 정보 (w)
            laserOdometry.pose.pose.position.x = t_last_curr_ndt.x(); // 현재 위치 (x)
            laserOdometry.pose.pose.position.y = t_last_curr_ndt.y(); // 현재 위치 (y)
            laserOdometry.pose.pose.position.z = t_last_curr_ndt.z(); // 현재 위치 (z)
            laserOdometry.twist.twist.linear.x = score; // 오도메트리 점수
            pubLaserOdometry.publish(laserOdometry);

            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header; // 레이저 포즈의 헤더 정보를 오도메트리 헤더로 설정
            laserPose.pose.position.x = t_w_curr.x(); // 레이저 포즈의 x 위치 설정
            laserPose.pose.position.y = t_w_curr.y(); // 레이저 포즈의 y 위치 설정
            laserPose.pose.position.z = t_w_curr.z(); // 레이저 포즈의 z 위치 설정
            laserPose.pose.orientation.x = q_w_curr.x(); // 레이저 포즈의 x 회전 정보 설정
            laserPose.pose.orientation.y = q_w_curr.y(); // 레이저 포즈의 y 회전 정보 설정
            laserPose.pose.orientation.z = q_w_curr.z(); // 레이저 포즈의 z 회전 정보 설정
            laserPose.pose.orientation.w = q_w_curr.w(); // 레이저 포즈의 w 회전 정보 설정
            laserPath.header.stamp = laserOdometry.header.stamp; // 레이저 경로 헤더의 타임스탬프 설정
            laserPath.poses.push_back(laserPose); // 레이저 경로에 레이저 포즈 추가
            laserPath.header.frame_id = "/map"; // 레이저 경로의 프레임 ID 설정
            pubLaserPath.publish(laserPath); // 레이저 경로 게시
            
            sensor_msgs::PointCloud2 ndtcloud;
            pcl::toROSMsg(*NDTPoints, ndtcloud); // NDT 포인트 클라우드를 ROS 메시지로 변환
            ndtcloud.header.frame_id = "map"; // NDT 클라우드의 프레임 ID 설정
            ndtcloud.header.stamp = ros::Time(timeNDT); // NDT 클라우드의 타임스탬프 설정
            pubNDTCloud.publish(ndtcloud); // NDT 클라우드 게시

        }

        rate.sleep();
    }
    return 0;
}

