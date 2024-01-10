#include <queue>
#include <mutex>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h> 

#include<time.h>
#include "e_loam/lidarFactor.hpp"

int corner_correspondence = 0; // 코너 포인트 간 일치 개수
int plane_correspondence = 0; // 평면 포인트 간 일치 개수
int texture_correspondence = 0; // 텍스처 포인트 간 일치 개수

// 코너 포인트와 평면 포인트에 대한 Kd 트리 생성
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

// 텍스처 포인트에 대한 Kd 트리 생성
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeTextureLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

// 코너 포인트 클라우드 초기화
pcl::PointCloud<pcl::PointXYZI>::Ptr EdgePointsSharp(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr EdgePointsLessSharp(new pcl::PointCloud<pcl::PointXYZI>());

// 평면 포인트 클라우드 초기화
pcl::PointCloud<pcl::PointXYZI>::Ptr PlanarPointsFlat(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr PlanarPointsLessFlat(new pcl::PointCloud<pcl::PointXYZI>());

// 텍스처 포인트 클라우드 초기화
pcl::PointCloud<pcl::PointXYZI>::Ptr TexturePointsSharp(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr TexturePointsLessSharp(new pcl::PointCloud<pcl::PointXYZI>());

// 코너 포인트 버퍼 초기화
std::queue<sensor_msgs::PointCloud2ConstPtr> EdgeSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> EdgeLessSharpBuf;

// 텍스처 포인트 버퍼 초기화
std::queue<sensor_msgs::PointCloud2ConstPtr> TextureSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> TextureLessSharpBuf;

// 평면 포인트 버퍼 초기화
std::queue<sensor_msgs::PointCloud2ConstPtr> PlanarFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> PlanarLessFlatBuf;

// 이전의 레이저 클라우드 (코너, 평면, 텍스처) 초기화
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudEdgeLast(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudPlanarLast(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudTextureLast(new pcl::PointCloud<pcl::PointXYZI>());

// 현재 프레임에서 월드 프레임으로의 변환 (쿼터니언과 위치 벡터)
Eigen::Quaterniond q_w_curr(1, 0, 0, 0); // 초기 쿼터니언 값 (단위 쿼터니언)
Eigen::Vector3d t_w_curr(0, 0, 0); // 초기 위치 벡터 (원점)

// 이전 프레임과 현재 프레임 사이의 변환 (쿼터니언과 위치 벡터)
double para_q[4] = {0, 0, 0, 1}; // 초기 쿼터니언 값 (단위 쿼터니언)
double para_t[3] = {0, 0, 0}; // 초기 위치 벡터 (원점)

// 다양한 시간 변수 초기화
double timeEdgePointsSharp = 0;
double timeEdgePointsLessSharp = 0;
double timeTexturePointsSharp = 0;
double timeTexturePointsLessSharp = 0;
double timePlanarPointsFlat = 0;
double timePlanarPointsLessFlat = 0;

// 변환 관련 변수 초기화
Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q); // 쿼터니언 맵핑
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t); // 위치 벡터 맵핑

// ROS 퍼블리셔 초기화
ros::Publisher pubLaserOdometry; // 레이저 오도메트리 퍼블리셔
ros::Publisher pubLaserPath; // 레이저 경로 퍼블리셔

// 레이저 경로 메시지 초기화
nav_msgs::Path laserPath;

// 시스템 초기화 여부
bool systemInited = false;

// 메시지 버퍼에 대한 뮤텍스 초기화
std::mutex mBuf;



// 리다 포인트를 왜곡 해제하는 함수
void TransformToStart(pcl::PointXYZI const *const pi, pcl::PointXYZI *const po)
{
    // 보간 비율 (s) 초기화
    double s = 1.0;

    // s = 1로 설정 (변환에 대한 선형 보간을 수행)

    // 현재 프레임과 이전 프레임 사이의 쿼터니언 보간
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);

    // 현재 프레임과 이전 프레임 사이의 위치 벡터 보간
    Eigen::Vector3d t_point_last = s * t_last_curr;

    // 입력된 리다 포인트의 위치 벡터 생성
    Eigen::Vector3d point(pi->x, pi->y, pi->z);

    // 보간된 변환을 사용하여 왜곡 해제된 포인트 생성
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    // 왜곡 해제된 포인트 정보를 출력 포인트에 저장
    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

// Sharp 에지 포인트 메시지를 받는 콜백 함수
void subEdgeSharpCallback(const sensor_msgs::PointCloud2ConstPtr& point_msgs)
{
    // 버퍼에 접근하기 위해 뮤텍스 잠금
    mBuf.lock();
    // 받은 메시지를 EdgeSharpBuf 버퍼에 저장
    EdgeSharpBuf.push(point_msgs);
    // 뮤텍스 잠금 해제
    mBuf.unlock();
}

// 덜 날카로운 에지 포인트 메시지를 받는 콜백 함수
void subEdgeLessSharpCallback(const sensor_msgs::PointCloud2ConstPtr& point_msgs)
{
    // 버퍼에 접근하기 위해 뮤텍스 잠금
    mBuf.lock();
    // 받은 메시지를 EdgeLessSharpBuf 버퍼에 저장
    EdgeLessSharpBuf.push(point_msgs);
    // 뮤텍스 잠금 해제
    mBuf.unlock();
}

// 선명한 텍스처 포인트 메시지를 받는 콜백 함수
void subTextureSharpCallback(const sensor_msgs::PointCloud2ConstPtr& point_msgs)
{
    // 버퍼에 접근하기 위해 뮤텍스 잠금
    mBuf.lock();
    // 받은 메시지를 TextureSharpBuf 버퍼에 저장
    TextureSharpBuf.push(point_msgs);
    // 뮤텍스 잠금 해제
    mBuf.unlock();
}

// 덜 선명한 텍스처 포인트 메시지를 받는 콜백 함수
void subTextureLessSharpCallback(const sensor_msgs::PointCloud2ConstPtr& point_msgs)
{
    // 버퍼에 접근하기 위해 뮤텍스 잠금
    mBuf.lock();
    // 받은 메시지를 TextureLessSharpBuf 버퍼에 저장
    TextureLessSharpBuf.push(point_msgs);
    // 뮤텍스 잠금 해제
    mBuf.unlock();
}


// 평면 모델의 평면 포인트 메시지를 받는 콜백 함수
void subPlanarFlatCallback(const sensor_msgs::PointCloud2ConstPtr& point_msgs)
{
    // 버퍼에 접근하기 위해 뮤텍스 잠금
    mBuf.lock();
    // 받은 메시지를 PlanarFlatBuf 버퍼에 저장
    PlanarFlatBuf.push(point_msgs);
    // 뮤텍스 잠금 해제
    mBuf.unlock();
}

// 덜 평면 모델의 평면 포인트 메시지를 받는 콜백 함수
void subPlanarLessFlatCallback(const sensor_msgs::PointCloud2ConstPtr& point_msgs)
{
    // 버퍼에 접근하기 위해 뮤텍스 잠금
    mBuf.lock();
    // 받은 메시지를 PlanarLessFlatBuf 버퍼에 저장
    PlanarLessFlatBuf.push(point_msgs);
    // 뮤텍스 잠금 해제
    mBuf.unlock();
}

int main(int argc, char** argv) {
    // ROS 노드 초기화
    ros::init(argc, argv, "test_odometry_loam_node");
    ros::NodeHandle node = ros::NodeHandle();

    // ROS 토픽을 구독하는 subscriber 설정
    ros::Subscriber subEdgePointsSharp = node.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, &subEdgeSharpCallback);
    ros::Subscriber subEdgePointsLessSharp = node.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, &subEdgeLessSharpCallback);
    ros::Subscriber subTexturePointsSharp = node.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_texture_sharp", 100, &subTextureSharpCallback);
    ros::Subscriber subTexturePointsLessSharp = node.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_texture_less_sharp", 100, &subTextureLessSharpCallback);
    ros::Subscriber subPlanarPointsFlat = node.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, &subPlanarFlatCallback);
    ros::Subscriber subPlanarPointsLessFlat = node.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, &subPlanarLessFlatCallback);

    // ROS 메시지를 발행하는 publisher 설정
    pubLaserOdometry = node.advertise<nav_msgs::Odometry>("/laser_odom_loam", 10);
    pubLaserPath = node.advertise<nav_msgs::Path>("/laser_odom_loam_path", 10);

    // 노드 루프 주기 설정
    ros::Rate rate(100);

    while (ros::ok()) {
        ros::spinOnce();
    
        // 모든 버퍼에 데이터가 들어왔을 때 처리
        if (!EdgeLessSharpBuf.empty() && !EdgeSharpBuf.empty() &&
            !TextureLessSharpBuf.empty() && !TextureSharpBuf.empty() &&
            !PlanarLessFlatBuf.empty() && !PlanarFlatBuf.empty()) {
            // 각 센서 데이터의 타임스탬프 값을 가져옴
            timeEdgePointsSharp = EdgeSharpBuf.front()->header.stamp.toSec();
            timeEdgePointsLessSharp = EdgeLessSharpBuf.front()->header.stamp.toSec();
            timeTexturePointsSharp = TextureSharpBuf.front()->header.stamp.toSec();
            timeTexturePointsLessSharp = TextureLessSharpBuf.front()->header.stamp.toSec();
            timePlanarPointsFlat = PlanarFlatBuf.front()->header.stamp.toSec();
            timePlanarPointsLessFlat = PlanarLessFlatBuf.front()->header.stamp.toSec();
    
            // 모든 센서 데이터의 타임스탬프가 동일한지 확인
            if (timeEdgePointsSharp != timeEdgePointsLessSharp || timeTexturePointsSharp != timeEdgePointsLessSharp ||
                timeTexturePointsLessSharp != timeEdgePointsLessSharp || timePlanarPointsFlat != timeEdgePointsLessSharp ||
                timePlanarPointsLessFlat != timeEdgePointsLessSharp) 
            {
                printf("unsync messeage!"); // 동기화되지 않은 메시지가 있을 경우 메시지 출력
                ROS_BREAK(); // ROS 루프 중단
            }


            // 버퍼 잠금 시작
            mBuf.lock();
            
            // EdgePointsSharp 버퍼 비우기
            EdgePointsSharp->clear();
            // ROS 메시지에서 PointCloud 데이터를 EdgePointsSharp에 변환
            pcl::fromROSMsg(*EdgeSharpBuf.front(), *EdgePointsSharp);
            // EdgeSharpBuf에서 데이터 삭제
            EdgeSharpBuf.pop();
            
            // EdgePointsLessSharp 버퍼 비우기
            EdgePointsLessSharp->clear();
            // ROS 메시지에서 PointCloud 데이터를 EdgePointsLessSharp에 변환
            pcl::fromROSMsg(*EdgeLessSharpBuf.front(), *EdgePointsLessSharp);
            // EdgeLessSharpBuf에서 데이터 삭제
            EdgeLessSharpBuf.pop();
            
            // TexturePointsSharp 버퍼 비우기
            TexturePointsSharp->clear();
            // ROS 메시지에서 PointCloud 데이터를 TexturePointsSharp에 변환
            pcl::fromROSMsg(*TextureSharpBuf.front(), *TexturePointsSharp);
            // TextureSharpBuf에서 데이터 삭제
            TextureSharpBuf.pop();
            
            // TexturePointsLessSharp 버퍼 비우기
            TexturePointsLessSharp->clear();
            // ROS 메시지에서 PointCloud 데이터를 TexturePointsLessSharp에 변환
            pcl::fromROSMsg(*TextureLessSharpBuf.front(), *TexturePointsLessSharp);
            // TextureLessSharpBuf에서 데이터 삭제
            TextureLessSharpBuf.pop();
            
            // PlanarPointsFlat 버퍼 비우기
            PlanarPointsFlat->clear();
            // ROS 메시지에서 PointCloud 데이터를 PlanarPointsFlat에 변환
            pcl::fromROSMsg(*PlanarFlatBuf.front(), *PlanarPointsFlat);
            // PlanarFlatBuf에서 데이터 삭제
            PlanarFlatBuf.pop();
                        
            // PlanarPointsLessFlat 버퍼 비우기
            PlanarPointsLessFlat->clear();
            // ROS 메시지에서 PointCloud 데이터를 PlanarPointsLessFlat에 변환
            pcl::fromROSMsg(*PlanarLessFlatBuf.front(), *PlanarPointsLessFlat);
            // PlanarLessFlatBuf에서 데이터 삭제
            PlanarLessFlatBuf.pop();
            
            // 버퍼 잠금 해제
            mBuf.unlock();
            
            // 현재 시간을 측정하여 t1 변수에 저장
            auto t1 = ros::WallTime::now();
            
            // 시스템이 초기화되지 않은 경우
            if (!systemInited)
            {
                // 시스템 초기화 완료 플래그를 true로 설정하고 메시지 출력
                systemInited = true;
                std::cout << "초기화 완료\n";
            }
            else
            {   
                // EdgePointsSharp의 포인트 개수를 cornerPointsSharpNum 변수에 저장
                int cornerPointsSharpNum = EdgePointsSharp->points.size();
            
                // PlanarPointsFlat의 포인트 개수를 surfPointsFlatNum 변수에 저장
                int surfPointsFlatNum = PlanarPointsFlat->points.size();
            
                // TexturePointsSharp의 포인트 개수를 texturePointsSharpNum 변수에 저장
                int texturePointsSharpNum = TexturePointsSharp->points.size();


                // 최적화를 두 번 반복하는 루프 (opti_counter가 0 또는 1인 경우)
                for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)
                {
                    // 각 반복마다 해당 변수를 초기화
                    corner_correspondence = 0;
                    plane_correspondence = 0;
                    texture_correspondence = 0;
                
                    // Ceres 라이브러리의 Huber 손실 함수를 사용한 손실 함수 설정
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                
                    // EigenQuaternionParameterization을 사용한 로컬 파라미터화 설정
                    ceres::LocalParameterization *q_parameterization =
                        new ceres::EigenQuaternionParameterization();
                
                    // ceres::Problem의 옵션 설정
                    ceres::Problem::Options problem_options;
                
                    // ceres::Problem 객체 생성
                    ceres::Problem problem(problem_options);
                
                    // 파라미터 블록 추가: para_q (4개의 파라미터), q_parameterization 사용
                    problem.AddParameterBlock(para_q, 4, q_parameterization);
                
                    // 파라미터 블록 추가: para_t (3개의 파라미터)
                    problem.AddParameterBlock(para_t, 3);
                
                    // pcl::PointXYZI 형식의 변수 pointSel 생성
                    pcl::PointXYZI pointSel;
                
                    // 포인트 검색을 위한 인덱스와 제곱 거리를 저장할 벡터 생성
                    std::vector<int> pointSearchInd;
                    std::vector<float> pointSearchSqDis;

                     // 코너 특징점에 대한 대응점 찾기
                    for (int i = 0; i < cornerPointsSharpNum; ++i)
                    {   
                        // 현재 코너 포인트를 시작점(기준점)으로 변환하여 pointSel에 저장
                        TransformToStart(&(EdgePointsSharp->points[i]), &pointSel);
                    
                        // k-d 트리를 사용하여 pointSel 주변에서 가장 가까운 포인트를 찾음
                        kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
                    
                        int closestPointInd = -1, minPointInd2 = -1;
                        if (pointSearchSqDis[0] < 25)
                        {
                            // 주변에 가까운 포인트가 존재하는 경우
                            closestPointInd = pointSearchInd[0];
                            int closestPointScanID = int(laserCloudEdgeLast->points[closestPointInd].intensity);
                    
                            double minPointSqDis2 = 25;
                    
                            // 주변에 더 가까운 포인트를 찾기 위해 스캔 라인 방향으로 검색
                            for (size_t j = closestPointInd + 1; j < laserCloudEdgeLast->points.size(); ++j)
                            {
                                // 같은 스캔 라인에 있는 경우, 계속 진행
                                if (int(laserCloudEdgeLast->points[j].intensity) <= closestPointScanID)
                                    continue;
                    
                                // 가까운 스캔이 아닌 경우, 루프 종료
                                if (int(laserCloudEdgeLast->points[j].intensity) > (closestPointScanID + 2.5))
                                    break;
                    
                                // 두 포인트 사이의 거리 제곱 계산
                                double pointSqDis = (laserCloudEdgeLast->points[j].x - pointSel.x) *
                                                    (laserCloudEdgeLast->points[j].x - pointSel.x) +
                                                    (laserCloudEdgeLast->points[j].y - pointSel.y) *
                                                    (laserCloudEdgeLast->points[j].y - pointSel.y) +
                                                    (laserCloudEdgeLast->points[j].z - pointSel.z) *
                                                    (laserCloudEdgeLast->points[j].z - pointSel.z);
                    
                                if (pointSqDis < minPointSqDis2)
                                {
                                    // 더 가까운 포인트를 찾음
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }


                            // 감소하는 스캔 라인 방향으로 검색
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // 같은 스캔 라인에 있는 경우, 계속 진행
                                if (int(laserCloudEdgeLast->points[j].intensity) >= closestPointScanID)
                                    continue;
                            
                                // 가까운 스캔이 아닌 경우, 루프 종료
                                if (int(laserCloudEdgeLast->points[j].intensity) < (closestPointScanID - 2.5))
                                    break;
                            
                                // 두 포인트 사이의 거리 제곱 계산
                                double pointSqDis = (laserCloudEdgeLast->points[j].x - pointSel.x) *
                                                    (laserCloudEdgeLast->points[j].x - pointSel.x) +
                                                    (laserCloudEdgeLast->points[j].y - pointSel.y) *
                                                    (laserCloudEdgeLast->points[j].y - pointSel.y) +
                                                    (laserCloudEdgeLast->points[j].z - pointSel.z) *
                                                    (laserCloudEdgeLast->points[j].z - pointSel.z);
                            
                                if (pointSqDis < minPointSqDis2)
                                {
                                    // 더 가까운 포인트를 찾음
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }

                        }
                        // 만약 minPointInd2가 유효한 경우 (both closestPointInd and minPointInd2가 유효한 경우)
                        if (minPointInd2 >= 0)
                        {
                            // 현재 코너 포인트의 위치를 Eigen 벡터로 저장
                            Eigen::Vector3d curr_point(EdgePointsSharp->points[i].x,
                                                       EdgePointsSharp->points[i].y,
                                                       EdgePointsSharp->points[i].z);
                        
                            // 가장 가까운 포인트의 위치를 Eigen 벡터로 저장
                            Eigen::Vector3d last_point_a(laserCloudEdgeLast->points[closestPointInd].x,
                                                        laserCloudEdgeLast->points[closestPointInd].y,
                                                        laserCloudEdgeLast->points[closestPointInd].z);
                        
                            // 두 번째로 가까운 포인트의 위치를 Eigen 벡터로 저장
                            Eigen::Vector3d last_point_b(laserCloudEdgeLast->points[minPointInd2].x,
                                                        laserCloudEdgeLast->points[minPointInd2].y,
                                                        laserCloudEdgeLast->points[minPointInd2].z);
                        
                            double s = 1.0;
                        
                            // LidarEdgeFactor 클래스를 사용하여 코스트 함수 생성
                            ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                        
                            // 문제에 코스트 함수 추가 (파라미터 블록과 손실 함수도 함께 지정)
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                        
                            // 코너 대응점 카운트 증가
                            corner_correspondence++;
                        }
                    }
                // 텍스처 코너 특징점에 대한 대응점 찾기
                for (int i = 0; i < texturePointsSharpNum; ++i)
                {   
                    // 현재 텍스처 코너 포인트를 시작점(기준점)으로 변환하여 pointSel에 저장
                    TransformToStart(&(TexturePointsSharp->points[i]), &pointSel);
                
                    // k-d 트리를 사용하여 pointSel 주변에서 가장 가까운 포인트를 찾음
                    kdtreeTextureLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
                
                    int closestPointInd = -1, minPointInd2 = -1;
                    if (pointSearchSqDis[0] < 25)
                    {
                        // 주변에 가까운 포인트가 존재하는 경우
                        closestPointInd = pointSearchInd[0];
                        int closestPointScanID = int(laserCloudTextureLast->points[closestPointInd].intensity);
                
                        double minPointSqDis2 = 25;
                
                        // 주변에 더 가까운 포인트를 찾기 위해 스캔 라인 방향으로 검색
                        for (size_t j = closestPointInd + 1; j < laserCloudTextureLast->points.size(); ++j)
                        {
                            // 같은 스캔 라인에 있는 경우, 계속 진행
                            if (int(laserCloudTextureLast->points[j].intensity) <= closestPointScanID)
                                continue;
                
                            // 가까운 스캔이 아닌 경우, 루프 종료
                            if (int(laserCloudTextureLast->points[j].intensity) > (closestPointScanID + 2.5))
                                break;
                
                            // 두 포인트 사이의 거리 제곱 계산
                            double pointSqDis = (laserCloudTextureLast->points[j].x - pointSel.x) *
                                                (laserCloudTextureLast->points[j].x - pointSel.x) +
                                                (laserCloudTextureLast->points[j].y - pointSel.y) *
                                                (laserCloudTextureLast->points[j].y - pointSel.y) +
                                                (laserCloudTextureLast->points[j].z - pointSel.z) *
                                                (laserCloudTextureLast->points[j].z - pointSel.z);
                
                            if (pointSqDis < minPointSqDis2)
                            {
                                // 더 가까운 포인트를 찾음
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        }
                        
                        // 감소하는 스캔 라인 방향으로 검색
                        for (int j = closestPointInd - 1; j >= 0; --j)
                        {
                            // 같은 스캔 라인에 있는 경우, 계속 진행
                            if (int(laserCloudTextureLast->points[j].intensity) >= closestPointScanID)
                                continue;
                        
                            // 가까운 스캔이 아닌 경우, 루프 종료
                            if (int(laserCloudTextureLast->points[j].intensity) < (closestPointScanID - 2.5))
                                break;
                        
                            // 두 포인트 사이의 거리 제곱 계산
                            double pointSqDis = (laserCloudTextureLast->points[j].x - pointSel.x) *
                                                (laserCloudTextureLast->points[j].x - pointSel.x) +
                                                (laserCloudTextureLast->points[j].y - pointSel.y) *
                                                (laserCloudTextureLast->points[j].y - pointSel.y) +
                                                (laserCloudTextureLast->points[j].z - pointSel.z) *
                                                (laserCloudTextureLast->points[j].z - pointSel.z);
                        
                            if (pointSqDis < minPointSqDis2)
                            {
                                // 더 가까운 포인트를 찾음
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        }

                        }
                    
                        // 만약 minPointInd2가 유효한 경우 (both closestPointInd and minPointInd2가 유효한 경우)
                        if (minPointInd2 >= 0)
                        {
                                // 현재 텍스처 코너 포인트의 위치를 Eigen 벡터로 저장
                                Eigen::Vector3d curr_point(TexturePointsSharp->points[i].x,
                                                            TexturePointsSharp->points[i].y,
                                                            TexturePointsSharp->points[i].z);
                            
                                // 가장 가까운 포인트의 위치를 Eigen 벡터로 저장
                                Eigen::Vector3d last_point_a(laserCloudTextureLast->points[closestPointInd].x,
                                                            laserCloudTextureLast->points[closestPointInd].y,
                                                            laserCloudTextureLast->points[closestPointInd].z);
                            
                                // 두 번째로 가까운 포인트의 위치를 Eigen 벡터로 저장
                                Eigen::Vector3d last_point_b(laserCloudTextureLast->points[minPointInd2].x,
                                                            laserCloudTextureLast->points[minPointInd2].y,
                                                            laserCloudTextureLast->points[minPointInd2].z);
                            
                                double s = 1.0;
                            
                                // LidarEdgeFactor 클래스를 사용하여 코스트 함수 생성
                                ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                            
                                // 문제에 코스트 함수 추가 (파라미터 블록과 손실 함수도 함께 지정)
                                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            
                                // 코너 대응점 카운트 증가
                                corner_correspondence++;
                        }

                    }

                    // 평면 특징점에 대한 대응점 찾기
                for (int i = 0; i < surfPointsFlatNum; ++i)
                {   
                    // 현재 평면 포인트를 시작점(기준점)으로 변환하여 pointSel에 저장
                    TransformToStart(&(PlanarPointsFlat->points[i]), &pointSel);
                
                    // k-d 트리를 사용하여 pointSel 주변에서 가장 가까운 포인트를 찾음
                    kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
                
                    int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                    if (pointSearchSqDis[0] < 25)
                    {
                        closestPointInd = pointSearchInd[0];
                
                        // 가장 가까운 포인트의 스캔 ID 가져오기
                        int closestPointScanID = int(laserCloudPlanarLast->points[closestPointInd].intensity);
                        double minPointSqDis2 = 25, minPointSqDis3 = 25;
                
                        // 주변 스캔 라인 방향으로 검색
                        for (size_t j = closestPointInd + 1; j < laserCloudPlanarLast->points.size(); ++j)
                        {
                            // 주변 스캔이 아닌 경우, 루프 종료
                            if (int(laserCloudPlanarLast->points[j].intensity) > (closestPointScanID + 2.5))
                                break;
                
                            double pointSqDis = (laserCloudPlanarLast->points[j].x - pointSel.x) *
                                                (laserCloudPlanarLast->points[j].x - pointSel.x) +
                                                (laserCloudPlanarLast->points[j].y - pointSel.y) *
                                                (laserCloudPlanarLast->points[j].y - pointSel.y) +
                                                (laserCloudPlanarLast->points[j].z - pointSel.z) *
                                                (laserCloudPlanarLast->points[j].z - pointSel.z);
                
                            // 동일 또는 낮은 스캔 라인에 있는 경우
                            if (int(laserCloudPlanarLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                            {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                            // 높은 스캔 라인에 있는 경우
                            else if (int(laserCloudPlanarLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                            {
                                minPointSqDis3 = pointSqDis;
                                minPointInd3 = j;
                            }
                        }

                            // 감소하는 스캔 라인 방향으로 검색
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // 주변 스캔이 아닌 경우, 루프 종료
                                if (int(laserCloudPlanarLast->points[j].intensity) < (closestPointScanID - 2.5))
                                    break;
                            
                                double pointSqDis = (laserCloudPlanarLast->points[j].x - pointSel.x) *
                                                    (laserCloudPlanarLast->points[j].x - pointSel.x) +
                                                    (laserCloudPlanarLast->points[j].y - pointSel.y) *
                                                    (laserCloudPlanarLast->points[j].y - pointSel.y) +
                                                    (laserCloudPlanarLast->points[j].z - pointSel.z) *
                                                    (laserCloudPlanarLast->points[j].z - pointSel.z);
                            
                                // 동일 또는 높은 스캔 라인에 있는 경우
                                if (int(laserCloudPlanarLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                else if (int(laserCloudPlanarLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    // 더 가까운 포인트를 찾음
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                        // minPointInd2와 minPointInd3가 모두 유효한 경우
                        if (minPointInd2 >= 0 && minPointInd3 >= 0)
                        {
                            // 현재 평면 포인트의 위치를 Eigen 벡터로 저장
                            Eigen::Vector3d curr_point(PlanarPointsFlat->points[i].x,
                                                        PlanarPointsFlat->points[i].y,
                                                        PlanarPointsFlat->points[i].z);
                        
                            // 가장 가까운 포인트의 위치를 Eigen 벡터로 저장
                            Eigen::Vector3d last_point_a(laserCloudPlanarLast->points[closestPointInd].x,
                                                        laserCloudPlanarLast->points[closestPointInd].y,
                                                        laserCloudPlanarLast->points[closestPointInd].z);
                        
                            // 두 번째로 가까운 포인트의 위치를 Eigen 벡터로 저장
                            Eigen::Vector3d last_point_b(laserCloudPlanarLast->points[minPointInd2].x,
                                                        laserCloudPlanarLast->points[minPointInd2].y,
                                                        laserCloudPlanarLast->points[minPointInd2].z);
                        
                            // 세 번째로 가까운 포인트의 위치를 Eigen 벡터로 저장
                            Eigen::Vector3d last_point_c(laserCloudPlanarLast->points[minPointInd3].x,
                                                        laserCloudPlanarLast->points[minPointInd3].y,
                                                        laserCloudPlanarLast->points[minPointInd3].z);
                        
                            double s = 1.0;
                        
                            // LidarPlaneFactor 클래스를 사용하여 코스트 함수 생성
                            ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                        
                            // 문제에 코스트 함수 추가 (파라미터 블록과 손실 함수도 함께 지정)
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                        
                            // 평면 대응점 카운트 증가
                            plane_correspondence++;
                        }
                        }
                    }

                    //printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);

                       // 만약 코너 대응점과 평면 대응점의 합이 10보다 작다면
                    if ((corner_correspondence + plane_correspondence) < 10)
                    {
                        printf("less correspondence!! *************************************************\n");
                    }
                    
                    // ceres 최적화 설정
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR; // 선형 솔버 유형 설정
                    options.max_num_iterations = 4; // 최대 반복 횟수 설정
                    options.minimizer_progress_to_stdout = false; // 최적화 진행 상황을 콘솔에 출력하지 않음
                    ceres::Solver::Summary summary; // 최적화 결과 요약 정보
                    ceres::Solve(options, &problem, &summary); // ceres 최적화 수행
                }
                    // 현재 위치 및 자세 업데이트
                    t_w_curr = t_w_curr + q_w_curr * t_last_curr;
                    q_w_curr = q_w_curr * q_last_curr;
            }

                
             auto t2 = ros::WallTime::now(); // 현재 시간 저장
    
            // lidar odometry_loam 실행 시간 출력 (밀리초 단위)
            std::cout << "lidar odometry_loam: " << (t2 - t1).toSec() * 1000 << "  ms" << std::endl;
            
            // PointCloud 데이터 교환을 위해 포인터 변수를 사용하여 교체
            pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudTemp = EdgePointsLessSharp;
            EdgePointsLessSharp = laserCloudEdgeLast;
            laserCloudEdgeLast = laserCloudTemp;
            
            laserCloudTemp = TexturePointsLessSharp;
            TexturePointsLessSharp = laserCloudTextureLast;
            laserCloudTextureLast = laserCloudTemp;
            
            laserCloudTemp = PlanarPointsLessFlat;
            PlanarPointsLessFlat = laserCloudPlanarLast;
            laserCloudPlanarLast = laserCloudTemp;
            
            // k-d 트리에 입력 데이터 업데이트
            kdtreeCornerLast->setInputCloud(laserCloudEdgeLast);
            kdtreeSurfLast->setInputCloud(laserCloudPlanarLast);
            kdtreeTextureLast->setInputCloud(laserCloudTextureLast);

            // Odometry 메시지 생성 및 발행
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "/odom"; // Odometry의 좌표 프레임 설정
            laserOdometry.child_frame_id = "/velodyne"; // 자식 좌표 프레임 설정
            laserOdometry.header.stamp = ros::Time::now(); // 현재 시간 설정
            laserOdometry.pose.pose.orientation.x = q_last_curr.x(); // 회전 정보 (x 축)
            laserOdometry.pose.pose.orientation.y = q_last_curr.y(); // 회전 정보 (y 축)
            laserOdometry.pose.pose.orientation.z = q_last_curr.z(); // 회전 정보 (z 축)
            laserOdometry.pose.pose.orientation.w = q_last_curr.w(); // 회전 정보 (w)
            laserOdometry.pose.pose.position.x = t_last_curr.x(); // 위치 정보 (x 축)
            laserOdometry.pose.pose.position.y = t_last_curr.y(); // 위치 정보 (y 축)
            laserOdometry.pose.pose.position.z = t_last_curr.z(); // 위치 정보 (z 축)
            pubLaserOdometry.publish(laserOdometry); // Odometry 메시지 발행
            
            // PoseStamped 메시지 생성 및 발행
            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header; // 헤더 정보 설정
            laserPath.header.stamp = laserOdometry.header.stamp; // 패스의 헤더 시간 설정
            laserPose.pose.position.x = t_w_curr.x(); // 월드 좌표계에서의 위치 정보 (x 축)
            laserPose.pose.position.y = t_w_curr.y(); // 월드 좌표계에서의 위치 정보 (y 축)
            laserPose.pose.position.z = t_w_curr.z(); // 월드 좌표계에서의 위치 정보 (z 축)
            laserPose.pose.orientation.x = q_w_curr.x(); // 월드 좌표계에서의 회전 정보 (x 축)
            laserPose.pose.orientation.y = q_w_curr.y(); // 월드 좌표계에서의 회전 정보 (y 축)
            laserPose.pose.orientation.z = q_w_curr.z(); // 월드 좌표계에서의 회전 정보 (z 축)
            laserPose.pose.orientation.w = q_w_curr.w(); // 월드 좌표계에서의 회전 정보 (w)
            laserPath.poses.push_back(laserPose); // 패스에 PoseStamped 메시지 추가
            laserPath.header.frame_id = "/map"; // 패스의 좌표 프레임 설정
            pubLaserPath.publish(laserPath); // 패스 메시지 발행

        }

        rate.sleep();
    }

    return 0;
}

