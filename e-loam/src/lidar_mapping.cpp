#include <ros/ros.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/ISAM2.h>

#include <Eigen/Dense>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

#include <pclomp/ndt_omp.h>

using namespace std;
using namespace gtsam;

struct PointXYZIRPYT
{
    PCL_ADD_POINT4D  // PCL의 4D 포인트 추가 매크로 (X, Y, Z, padding)
    PCL_ADD_INTENSITY;  // 강도(Intensity)를 추가하는 PCL 매크로 (XYZ에 대한 선호된 방법)
    double q_x;  // 쿼터니언 x 요소
    double q_y;  // 쿼터니언 y 요소
    double q_z;  // 쿼터니언 z 요소
    double q_w;  // 쿼터니언 w 요소
    double time;  // 시간 정보
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // 새로운 할당자가 정확한 메모리 정렬을 가지도록 만듦
} EIGEN_ALIGN16;  // SSE 패딩을 강제하여 올바른 메모리 정렬 보장

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (double, q_x, q_x) (double, q_y, q_y) (double, q_z, q_z) (double, q_w, q_w)
                                   (double, time, time))
// PointXYZIRPYT 구조체를 PCL 포인트 클라우드 레지스트리에 등록
typedef PointXYZIRPYT PointTypePose;  // PointTypePose로 PointXYZIRPYT 구조체를 typedef
typedef pcl::PointXYZI PointType;  // pcl::PointXYZI를 PointType으로 typedef

NonlinearFactorGraph gtSAMgraph;
// gtSAMgraph: 그래프 기반 최적화를 위한 비선형 팩터 그래프 객체입니다.

Values initialEstimate;
// initialEstimate: 변수들에 대한 초기 추정치를 저장하는 객체입니다.

Values optimizedEstimate;
// optimizedEstimate: 최적화된 변수들의 추정치를 저장하는 객체입니다.

ISAM2 *isam;
// isam: iSAM2 (Incremental Smoothing and Mapping) 알고리즘을 사용하기 위한 포인터입니다.

Values isamCurrentEstimate;
// isamCurrentEstimate: iSAM2 알고리즘을 사용하여 현재 추정된 변수들의 값을 저장하는 객체입니다.

noiseModel::Diagonal::shared_ptr priorNoise;
// priorNoise: 대각선 형태의 노이즈 모델을 사용하는데, 사전 정보를 나타내는 노이즈 모델 객체입니다.

noiseModel::Diagonal::shared_ptr odometryNoise;
// odometryNoise: 대각선 형태의 노이즈 모델을 사용하는데, 오도메트리 정보를 나타내는 노이즈 모델 객체입니다.

noiseModel::Diagonal::shared_ptr constraintNoise;
// constraintNoise: 대각선 형태의 노이즈 모델을 사용하는데, 제약 조건 정보를 나타내는 노이즈 모델 객체입니다.


nav_msgs::Odometry odomAftMapped;
// odomAftMapped: 매핑된 이후의 오도메트리 정보를 저장하는 객체입니다.

vector<pcl::PointCloud<PointType>::Ptr> NDTCloudKeyFrames;
// NDTCloudKeyFrames: 포인트 클라우드를 저장하는 벡터입니다. 각 원소는 PointType 타입의 포인트 클라우드를 가리키는 포인터입니다.

vector<int> surroundingExistingKeyPoseID;
// surroundingExistingKeyPoseID: 주변에 존재하는 키포즈(ID)를 저장하는 벡터입니다.

deque<pcl::PointCloud<PointType>::Ptr> surroundingNDTCloudFrames;
// surroundingNDTCloudFrames: 주변 포인트 클라우드 프레임을 저장하는 덱(deque)입니다.
deque<pcl::PointCloud<PointType>::Ptr> recentNDTCloudKeyFrames;
// recentNDTCloudKeyFrames: 최근 포인트 클라우드 키포즈 프레임을 저장하는 덱(deque)입니다.
pcl::PointCloud<PointType>::Ptr laserCloudNDTLast;
// laserCloudNDTLast: NDT(Normal Distributions Transform)를 사용한 라이다 포인트 클라우드 정보를 저장하는 객체입니다.
pcl::PointCloud<PointType>::Ptr laserCloudNDTFromMap;
// laserCloudNDTFromMap: 지도로부터 추출한 NDT 포인트 클라우드 정보를 저장하는 객체입니다.
pcl::PointCloud<PointType>::Ptr laserCloudNDTFromMapDS;
// laserCloudNDTFromMapDS: 다운샘플링된(laserCloudNDTFromMap의) NDT 포인트 클라우드 정보를 저장하는 객체입니다.
pcl::PointCloud<PointType>::Ptr nearHistoryNDTKeyFrameCloud;
// nearHistoryNDTKeyFrameCloud: 근접한 역사적 NDT 키포즈 클라우드 정보를 저장하는 객체입니다.
pcl::PointCloud<PointType>::Ptr nearHistoryNDTKeyFrameCloudDS;
// nearHistoryNDTKeyFrameCloudDS: 다운샘플링된(nearHistoryNDTKeyFrameCloud의) 근접한 역사적 NDT 키포즈 클라우드 정보를 저장하는 객체입니다.
pcl::PointCloud<PointType>::Ptr latestNDTKeyFrameCloud;
// latestNDTKeyFrameCloud: 가장 최근 NDT 키포즈 클라우드 정보를 저장하는 객체입니다.

// PointTypePose의 XYZI는 cloudKeyPoses3D와 동일한 내용을 저장하며, RPY 각도 및 시간 값을 포함합니다.
pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
// cloudKeyPoses3D: 3D 포인트 키포즈 정보를 저장하는 객체입니다.
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
// cloudKeyPoses6D: 6D 포인트 키포즈 정보를 저장하는 객체입니다.
pcl::PointCloud<PointType>::Ptr surroundingKeyPoses;
// surroundingKeyPoses: 주변 포인트 키포즈 정보를 저장하는 객체입니다.
pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS;
// surroundingKeyPosesDS: 다운샘플링된(surroundingKeyPoses의) 주변 포인트 키포즈 정보를 저장하는 객체입니다.

pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
// kdtreeSurroundingKeyPoses: 주변 포인트 키포즈에 대한 Kd 트리(효율적인 탐색을 위한 데이터 구조)를 나타내는 객체입니다.
pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;
// kdtreeHistoryKeyPoses: 역사적 포인트 키포즈에 대한 Kd 트리를 나타내는 객체입니다.



pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // 전역 지도 시각화를 위한 다운샘플링 필터
pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // 전역 지도 시각화를 위한 다운샘플링 필터
pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // 주변 키포즈 시각화를 위한 다운샘플링 필터
pcl::VoxelGrid<PointType> downSizeFilterHistoryKeyFrames; // 역사적 키포즈 시각화를 위한 다운샘플링 필터
pcl::VoxelGrid<PointType> downSizeFilterNDT; // NDT 포인트 클라우드 다운샘플링을 위한 필터

pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap; // 전역 지도에 대한 Kd 트리
pcl::PointCloud<PointType>::Ptr globalMapKeyPoses; // 전역 지도의 키포즈 정보
pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS; // 다운샘플링된 전역 지도의 키포즈 정보
pcl::PointCloud<PointType>::Ptr globalMapKeyFrames; // 전역 지도의 키프레임 정보
pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS; // 다운샘플링된 전역 지도의 키프레임 정보

PointType previousRobotPosPoint; // 이전 로봇 위치 포인트
PointType currentRobotPosPoint; // 현재 로봇 위치 포인트

ros::Publisher pubLaserCloudSurround; // 주변 포인트 클라우드를 게시하는 ROS 퍼블리셔
ros::Publisher pubOdomAftMappedHighFrec; // 매핑된 오도메트리 정보를 게시하는 ROS 퍼블리셔
ros::Publisher pubLaserPath; // 레이저 경로를 게시하는 ROS 퍼블리셔

nav_msgs::Path laserPath; // 레이저 경로 정보를 저장하는 ROS 메시지

// 정규 분포 변환(NDT)을 위한 등록 객체 초기화
pcl::Registration<pcl::PointXYZI, pcl::PointXYZI>::Ptr registration;


vector<int> pointSearchInd;
// pointSearchInd: 포인트 검색 인덱스를 저장하는 벡터

vector<float> pointSearchSqDis;
// pointSearchSqDis: 포인트 검색 제곱 거리를 저장하는 벡터

vector<int> surroundingExistingKeyPosesID;
// surroundingExistingKeyPosesID: 주변에 존재하는 키포즈(ID)를 저장하는 벡터

double timeLaserCloudNDTLast;
// timeLaserCloudNDTLast: NDT 포인트 클라우드의 최근 시간 정보

double timeLaserOdometry;
// timeLaserOdometry: 라이다 오도메트리의 시간 정보

double timeLastProcessing;
// timeLastProcessing: 최근 데이터 처리 시간 정보

double odometry_weight = 0.0;
// odometry_weight: 오도메트리 가중치를 초기화하고 0.0으로 설정

int laserCloudNDTFromMapDSNum;
// laserCloudNDTFromMapDSNum: 다운샘플링된 NDT 포인트 클라우드의 포인트 수

Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);
// q_wmap_wodom, t_wmap_wodom: 월드 좌표계에서 오도메트리 좌표계로의 변환 행렬과 변환 벡터

Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);
// q_wodom_curr, t_wodom_curr: 이전 오도메트리 좌표계에서 현재 오도메트리 좌표계로의 변환 행렬과 변환 벡터

Eigen::Quaterniond q_wodom_curr_loam(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr_loam(0, 0, 0);
// q_wodom_curr_loam, t_wodom_curr_loam: LOAM에서 사용하는 현재 오도메트리 좌표계로의 변환 행렬과 변환 벡터

Eigen::Quaterniond q_wodom_curr_ndt(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr_ndt(0, 0, 0);
// q_wodom_curr_ndt, t_wodom_curr_ndt: NDT에서 사용하는 현재 오도메트리 좌표계로의 변환 행렬과 변환 벡터

Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);
// q_w_curr, t_w_curr: 현재 월드 좌표계에서 로봇 좌표계로의 변환 행렬과 변환 벡터

Eigen::Quaterniond q_w_curr_last(1, 0, 0, 0);
Eigen::Vector3d t_w_curr_last(0, 0, 0);
// q_w_curr_last, t_w_curr_last: 이전 월드 좌표계에서 로봇 좌표계로의 변환 행렬과 변환 벡터


size_t surroundingKeyframeSearchNum = 20;
// surroundingKeyframeSearchNum: 주변 키프레임 검색에 사용되는 키프레임 수

int latestFrameID = 0;
// latestFrameID: 가장 최근 프레임의 ID

int closestHistoryFrameID;
// closestHistoryFrameID: 가장 가까운 역사 프레임의 ID

int latestFrameIDLoopCloure;
// latestFrameIDLoopCloure: 루프 클로저를 위한 가장 최근 프레임의 ID

bool newLaserCloudNDTLast = false;
// newLaserCloudNDTLast: 새로운 NDT 포인트 클라우드가 있는지 여부를 나타내는 불리언 값

bool newLaserOdometry = false;
// newLaserOdometry: 새로운 라이다 오도메트리 정보가 있는지 여부를 나타내는 불리언 값

bool newLaserOdometry_ndt = false;
// newLaserOdometry_ndt: NDT를 사용한 새로운 라이다 오도메트리 정보가 있는지 여부를 나타내는 불리언 값

bool newLaserOdometry_loam = false;
// newLaserOdometry_loam: LOAM을 사용한 새로운 라이다 오도메트리 정보가 있는지 여부를 나타내는 불리언 값

bool loopClosureEnableFlag = true;
// loopClosureEnableFlag: 루프 클로저 활성화 플래그

bool potentialLoopFlag = false;
// potentialLoopFlag: 잠재적인 루프 클로저가 있음을 나타내는 불리언 값

bool aLoopIsClosed = false;
// aLoopIsClosed: 루프가 닫혔음을 나타내는 불리언 값

std::mutex mtx;
// mtx: 스레드 간 동기화를 위한 뮤텍스(mutex) 객체


void ndtCloudCallback(const sensor_msgs::PointCloud2ConstPtr& pointMsg) {
    // 라이다 포인트 클라우드 메시지의 타임스탬프를 timeLaserCloudNDTLast에 저장
    timeLaserCloudNDTLast = pointMsg->header.stamp.toSec();
    
    // laserCloudNDTLast 클라우드를 초기화
    laserCloudNDTLast->clear();
    
    // ROS 메시지 형식의 포인트 클라우드 데이터를 pcl::PointCloud 형식으로 변환하여 laserCloudNDTLast에 저장
    pcl::fromROSMsg(*pointMsg, *laserCloudNDTLast);
    
    // 새로운 NDT 포인트 클라우드가 수신되었음을 나타내는 플래그를 설정
    newLaserCloudNDTLast = true;
}


void laserOdometryNDTCallback(const nav_msgs::Odometry::ConstPtr& laserOdometryMSG) {
    // 라이다 오도메트리 메시지의 시간 정보를 timeLaserOdometry에 저장
    // (주석 처리된 부분은 시간 정보를 저장하는 것으로 보이지만 현재 주석 처리되어 있습니다)
    // timeLaserOdometry = laserOdometryMSG->header.stamp.toSec();
    
    // 라이다 오도메트리 메시지에서 자세 정보(Quaternion)를 q_wodom_curr_ndt에 저장
    q_wodom_curr_ndt.x() = laserOdometryMSG->pose.pose.orientation.x;
    q_wodom_curr_ndt.y() = laserOdometryMSG->pose.pose.orientation.y;
    q_wodom_curr_ndt.z() = laserOdometryMSG->pose.pose.orientation.z;
    q_wodom_curr_ndt.w() = laserOdometryMSG->pose.pose.orientation.w;
    
    // 라이다 오도메트리 메시지에서 위치 정보를 t_wodom_curr_ndt에 저장
    t_wodom_curr_ndt.x() = laserOdometryMSG->pose.pose.position.x;
    t_wodom_curr_ndt.y() = laserOdometryMSG->pose.pose.position.y;
    t_wodom_curr_ndt.z() = laserOdometryMSG->pose.pose.position.z;

    // 라이다 오도메트리 메시지에서 선속도 정보를 odometry_weight에 저장
    odometry_weight = laserOdometryMSG->twist.twist.linear.x;

    // 새로운 NDT 라이다 오도메트리 정보가 수신되었음을 나타내는 플래그를 설정
    newLaserOdometry_ndt = true;
}


void laserOdometryLoamCallback(const nav_msgs::Odometry::ConstPtr& laserOdometryMSG) {
    // laserOdometryMSG로부터 시간 정보를 가져옵니다.
    timeLaserOdometry = laserOdometryMSG->header.stamp.toSec();

    // laserOdometryMSG로부터 자세 정보(Quaternion)를 가져와서 적절한 변수에 할당합니다.
    q_wodom_curr_loam.x() = laserOdometryMSG->pose.pose.orientation.x;
    q_wodom_curr_loam.y() = laserOdometryMSG->pose.pose.orientation.y;
    q_wodom_curr_loam.z() = laserOdometryMSG->pose.pose.orientation.z;
    q_wodom_curr_loam.w() = laserOdometryMSG->pose.pose.orientation.w;

    // laserOdometryMSG로부터 위치 정보(Vector3)를 가져와서 적절한 변수에 할당합니다.
    t_wodom_curr_loam.x() = laserOdometryMSG->pose.pose.position.x;
    t_wodom_curr_loam.y() = laserOdometryMSG->pose.pose.position.y;
    t_wodom_curr_loam.z() = laserOdometryMSG->pose.pose.position.z;

    // 주석 처리된 코드 부분은 현재 비활성화되어 있습니다.
    // 이 부분은 Eigen 라이브러리를 사용하여 새로운 데이터를 계산하고 발행하는 부분입니다.
    // 주석을 해제하면 "odomAftMappedHighFrec" 및 "laserPath" 메시지를 발행할 수 있습니다.

    // Eigen::Quaterniond q_w_curr_tmp = q_wmap_wodom * q_wodom_curr;
    // Eigen::Vector3d t_w_curr_tmp = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;

    // nav_msgs::Odometry odomAftMapped;
    // odomAftMapped.header.frame_id = "/map";
    // odomAftMapped.child_frame_id = "/aft_mapped";
    // odomAftMapped.header.stamp = laserOdometryMSG->header.stamp;
    // odomAftMapped.pose.pose.orientation.x = q_w_curr_tmp.x();
    // odomAftMapped.pose.pose.orientation.y = q_w_curr_tmp.y();
    // odomAftMapped.pose.pose.orientation.z = q_w_curr_tmp.z();
    // odomAftMapped.pose.pose.orientation.w = q_w_curr_tmp.w();
    // odomAftMapped.pose.pose.position.x = t_w_curr_tmp.x();
    // odomAftMapped.pose.pose.position.y = t_w_curr_tmp.y();
    // odomAftMapped.pose.pose.position.z = t_w_curr_tmp.z();
    // pubOdomAftMappedHighFrec.publish(odomAftMapped);

    // geometry_msgs::PoseStamped laserPose;
    // laserPose.header = odomAftMapped.header;
    // laserPose.pose = odomAftMapped.pose.pose;
    // laserPath.header.stamp = odomAftMapped.header.stamp;
    // laserPath.poses.push_back(laserPose);
    // laserPath.header.frame_id = "/map";
    // pubLaserPath.publish(laserPath);

    // 새로운 레이저 오도메트리 데이터가 도착했음을 표시합니다.
    newLaserOdometry_loam = true;
}

// 초기 추정값 설정
void transformAssociateToMap()
{
    // 현재 로봇의 자세를 전역 좌표계로 변환합니다.
    q_w_curr = q_wmap_wodom * q_wodom_curr; // 현재 자세를 변환
    t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom; // 현재 위치를 변환
}

// 변환 업데이트 함수
void transformUpdate()
{
    // 로봇의 전역 자세 및 위치를 오도메트리 정보를 사용하여 업데이트합니다.
    q_wmap_wodom = q_w_curr * q_wodom_curr.inverse(); // 전역 자세 업데이트
    t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr; // 전역 위치 업데이트
}

// PCL(Point Cloud Library) 포인트를 gtsam의 Pose3로 변환하는 함수
Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
{
    // PCL 포인트를 gtsam의 Pose3로 변환합니다.
    return Pose3(
        Rot3::Quaternion(thisPoint.q_w, thisPoint.q_x, thisPoint.q_y, thisPoint.q_z), // 회전 정보
        Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)) // 위치 정보
    );
}


// PCL 포인트를 Affine3f 형태로 변환하는 함수
Eigen::Affine3f pclPointToAffine3fLidar(PointTypePose thisPoint){

    // 주어진 PCL 포인트의 회전과 위치 정보를 사용하여 Eigen::Isometry3d 객체 생성
    Eigen::Quaterniond tmp_q(thisPoint.q_w, thisPoint.q_x, thisPoint.q_y, thisPoint.q_z);
    Eigen::Vector3d tmp_t(thisPoint.x, thisPoint.y, thisPoint.z);
    Eigen::Isometry3d tmp_T = Eigen::Isometry3d::Identity();
    tmp_T.rotate(tmp_q); // 회전 정보 설정
    tmp_T.pretranslate(tmp_t); // 위치 정보 설정

    // Eigen::Isometry3d를 Eigen::Affine3f로 형변환하여 반환
    return Eigen::Affine3f(tmp_T.matrix().cast<float>());
}

// PCL 포인트 클라우드를 주어진 변환을 사용하여 변환하는 함수
pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn){
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    PointType point_To;

    int cloudSize = cloudIn->points.size();
    cloudOut->resize(cloudSize);

    // 주어진 변환을 사용하여 클라우드의 각 포인트를 변환합니다.
    Eigen::Quaterniond tmp_q(transformIn->q_w, transformIn->q_x, transformIn->q_y, transformIn->q_z);
    Eigen::Vector3d tmp_t(transformIn->x, transformIn->y, transformIn->z);
    for (int i = 0; i < cloudSize; ++i){

        // 클라우드의 각 포인트를 주어진 변환을 사용하여 새로운 위치로 변환합니다.
        Eigen::Vector3d pointFrom(cloudIn->points[i].x, cloudIn->points[i].y, cloudIn->points[i].z);
        Eigen::Vector3d pointTo =  tmp_q * pointFrom + tmp_t;

        point_To.x = pointTo.x();
        point_To.y = pointTo.y();
        point_To.z = pointTo.z();
        point_To.intensity = cloudIn->points[i].intensity;

        // 변환된 포인트를 새로운 클라우드에 저장합니다.
        cloudOut->points[i] = point_To;
    }
    return cloudOut;
}

void extractSurroundingKeyFrames(){

    // 만약 cloudKeyPoses3D가 비어있다면, 함수 실행 중지
    if (cloudKeyPoses3D->points.empty() == true)
        return;	

    // loopClosureEnableFlag 변수는 루프 클로저와 관련된 기능을 사용하는 경우에만 사용됩니다.

    if (loopClosureEnableFlag == true){

        // recentNDTCloudKeyFrames 리스트에 저장된 포인트 클라우드의 수가 surroundingKeyframeSearchNum보다 작다면,
        // 리스트를 비우고 새로운 포인트 클라우드를 채웁니다.
        if (recentNDTCloudKeyFrames.size() < surroundingKeyframeSearchNum){
            recentNDTCloudKeyFrames.clear();
            int numPoses = cloudKeyPoses3D->points.size();
            for (int i = numPoses-1; i >= 0; --i){
                // cloudKeyPoses3D의 intensity 필드에는 포인트 클라우드의 인덱스 정보가 저장되어 있습니다.
                // 저장된 인덱스는 1부터 시작합니다.
                int thisKeyInd = (int)cloudKeyPoses3D->points[i].intensity;
                PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
                // 현재 포인트 클라우드를 주어진 변환을 사용하여 변환한 후, recentNDTCloudKeyFrames 리스트에 추가합니다.
                recentNDTCloudKeyFrames.push_front(transformPointCloud(NDTCloudKeyFrames[thisKeyInd], &thisTransformation));
                if (recentNDTCloudKeyFrames.size() >= surroundingKeyframeSearchNum)
                    break;
            }
	
	       }else{
	    // recentCornerCloudKeyFrames 리스트에 저장된 포인트 클라우드가 많을 경우,
	    // 가장 오래된 포인트 클라우드를 리스트에서 제거하고, 최신의 포인트 클라우드를 추가합니다.
	    if (latestFrameID != cloudKeyPoses3D->points.size() - 1){
	        recentNDTCloudKeyFrames.pop_front();
	        
	        latestFrameID = cloudKeyPoses3D->points.size() - 1;
	        PointTypePose thisTransformation = cloudKeyPoses6D->points[latestFrameID];
	
	        recentNDTCloudKeyFrames.push_back(transformPointCloud(NDTCloudKeyFrames[latestFrameID], &thisTransformation));
	    }
	}
	
	// recentNDTCloudKeyFrames에 저장된 모든 포인트 클라우드를 합칩니다.
	for (int i = 0; i < recentNDTCloudKeyFrames.size(); ++i){
	    *laserCloudNDTFromMap += *recentNDTCloudKeyFrames[i];
	}


   } else{ 
    // 루프 클로저 사용하지 않을 때,

    // 주변의 키프레임 리스트와 다운샘플링된 버전을 초기화합니다.
    surroundingKeyPoses->clear();
    surroundingKeyPosesDS->clear();

    // kdtreeSurroundingKeyPoses에 cloudKeyPoses3D를 입력으로 설정합니다.
    kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);

    // currentRobotPosPoint를 중심으로 6.0 반경 내의 포즈를 검색합니다.
    kdtreeSurroundingKeyPoses->radiusSearch(currentRobotPosPoint, 6.0, pointSearchInd, pointSearchSqDis, 0);
    for (size_t i = 0; i < pointSearchInd.size(); ++i)
        // 검색된 포즈를 surroundingKeyPoses 리스트에 추가합니다.
        surroundingKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchInd[i]]);

    // 다운샘플링 필터를 사용하여 surroundingKeyPoses 리스트를 filtering하고, 결과를 surroundingKeyPosesDS에 저장합니다.
    downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
    downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

        size_t numSurroundingPosesDS = surroundingKeyPosesDS->points.size();

	// surroundingExistingKeyPosesID와 surroundingKeyPosesDS를 비교하여 중복되는 포즈를 확인합니다.
	for (size_t i = 0; i < surroundingExistingKeyPosesID.size(); ++i) {
	    bool existingFlag = false;
	    for (size_t j = 0; j < numSurroundingPosesDS; ++j){
		// 중복을 확인하기 위해 surroundingExistingKeyPosesID[i]와 surroundingKeyPosesDS의 포즈 인덱스를 비교합니다.
		// 중복되는 경우, existingFlag를 true로 설정하고 루프를 빠져나옵니다.
		if (surroundingExistingKeyPosesID[i] == (int)surroundingKeyPosesDS->points[j].intensity){
		    existingFlag = true;
		    break;
		}
	    }
	    
	    if (existingFlag == false){
		// 만약 existingFlag가 false이면, surroundingExistingKeyPosesID[i]가 중복되지 않았다는 뜻입니다.
		// 따라서 해당 포즈를 제거하고, 관련된 데이터를 리스트에서도 제거합니다.
		surroundingExistingKeyPosesID.erase(surroundingExistingKeyPosesID.begin() + i);
		surroundingNDTCloudFrames.erase(surroundingNDTCloudFrames.begin() + i);
		--i; // 리스트에서 제거된 항목으로 인해 i를 다시 검토해야 합니다.
	    }
	}


	// 이전의 두 번째 이중 for 루프는 데이터 삭제에 사용되었으며, 이제는 데이터를 추가하는 데 사용됩니다.
	
	// numSurroundingPosesDS 개수만큼 반복하여 새로운 포즈를 추가합니다.
	for (size_t i = 0; i < numSurroundingPosesDS; ++i) {
	    bool existingFlag = false;
	
	    // surroundingExistingKeyPosesID에 있는 각 포즈 인덱스와 비교하여 중복 여부를 확인합니다.
	    for (auto iter = surroundingExistingKeyPosesID.begin(); iter != surroundingExistingKeyPosesID.end(); ++iter){
	        // 중복되는 경우 existingFlag를 true로 설정하고 루프를 종료합니다.
	        if ((*iter) == (int)surroundingKeyPosesDS->points[i].intensity){
	            existingFlag = true;
	            break;
	        }
	    }
	    
	    if (existingFlag == true){
	        // 이미 존재하는 포즈인 경우, 다음 반복으로 진행합니다.
	        continue;
	    } else {
	        // 존재하지 않는 포즈인 경우, 해당 포즈의 인덱스를 가져오고,
	        // 이를 사용하여 해당 포즈를 cloudKeyPoses6D에서 가져옵니다.
	        int thisKeyInd = (int)surroundingKeyPosesDS->points[i].intensity;
	        PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
	
	        // surroundingExistingKeyPosesID와 surroundingNDTCloudFrames에 새로운 포즈를 추가합니다.
	        surroundingExistingKeyPosesID.push_back(thisKeyInd);
	        surroundingNDTCloudFrames.push_back(transformPointCloud(NDTCloudKeyFrames[thisKeyInd], &thisTransformation));
	    }
	}
	
	// 추가된 포즈들을 laserCloudNDTFromMap에 더해줍니다.
	for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i) {
	    *laserCloudNDTFromMap += *surroundingNDTCloudFrames[i];
	}
    }

	// 다운샘플링 필터에 laserCloudNDTFromMap을 입력으로 설정합니다.
	downSizeFilterNDT.setInputCloud(laserCloudNDTFromMap);
	
	// 필터를 적용하여 다운샘플링된 결과를 laserCloudNDTFromMapDS에 저장합니다.
	downSizeFilterNDT.filter(*laserCloudNDTFromMapDS);
	
	// 다운샘플링된 포인트 클라우드의 포인트 수를 저장합니다.
	laserCloudNDTFromMapDSNum = laserCloudNDTFromMapDS->points.size();

}

void ndtRegistration() {

    // 만약 다운샘플링된 포인트 클라우드가 비어 있다면 함수를 종료합니다.
    if(laserCloudNDTFromMapDS->points.empty()) 
        return;
    
    // 등록 프로세스에 타겟 포인트 클라우드(laserCloudNDTFromMapDS)와 소스 포인트 클라우드(laserCloudNDTLast)를 설정합니다.
    registration->setInputTarget(laserCloudNDTFromMapDS);
    registration->setInputSource(laserCloudNDTLast);

    // 등록된 포인트 클라우드를 저장할 변수를 선언합니다.
    pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>());

    // 초기 추정값을 설정합니다. 이 예에서는 현재 위치와 회전을 사용하여 초기 추정값을 설정합니다.
    Eigen::Matrix4d init_guess;
    init_guess.setIdentity();
    init_guess.block<3,3>(0,0) = q_w_curr.toRotationMatrix(); // 회전 정보 설정
    init_guess.topRightCorner(3,1) = t_w_curr; // 위치 정보 설정

    // NDT 등록을 수행합니다. 결과는 "aligned" 포인트 클라우드에 저장됩니다.
    registration->align(*aligned, init_guess.cast<float>());

    // NDT 등록 결과에서 최종 변환 정보를 추출합니다.
    Eigen::Quaternionf tmp_q(registration->getFinalTransformation().topLeftCorner<3, 3>());
    Eigen::Vector3f tmp_t(registration->getFinalTransformation().topRightCorner<3, 1>());
    q_w_curr = tmp_q.cast<double>();
    t_w_curr = tmp_t.cast<double>();

    // 변환 업데이트를 수행합니다.
    transformUpdate();
}


void saveKeyFramesAndFactor(){

    // 현재 로봇의 위치 정보를 업데이트합니다.
    currentRobotPosPoint.x = t_w_curr.x();
    currentRobotPosPoint.y = t_w_curr.y();
    currentRobotPosPoint.z = t_w_curr.z();

    // 키프레임을 저장해야 하는지 여부를 결정합니다.
    bool saveThisKeyFrame = true;

    // 이전 로봇 위치와 현재 로봇 위치 사이의 거리를 계산하여, 일정 거리 이상일 경우에만 키프레임을 저장합니다.
    if (sqrt((previousRobotPosPoint.x-currentRobotPosPoint.x)*(previousRobotPosPoint.x-currentRobotPosPoint.x)
            +(previousRobotPosPoint.y-currentRobotPosPoint.y)*(previousRobotPosPoint.y-currentRobotPosPoint.y)
            +(previousRobotPosPoint.z-currentRobotPosPoint.z)*(previousRobotPosPoint.z-currentRobotPosPoint.z)) < 0.3){
        saveThisKeyFrame = false;
    }

    // 키프레임을 저장하지 않을 경우, 이전 키프레임이 존재하면 함수를 종료합니다.
    if (saveThisKeyFrame == false && !cloudKeyPoses3D->points.empty())
        return;

    // 키프레임을 저장할 경우, 현재 로봇 위치 정보를 업데이트하고, 이전 로봇 위치를 현재 위치로 설정합니다.
    previousRobotPosPoint = currentRobotPosPoint;

    if (cloudKeyPoses3D->points.empty()){
        // 최초 키프레임인 경우, 그래프에 초기 위치 및 회전 정보를 추가합니다.
        gtSAMgraph.add(PriorFactor<Pose3>(0, Pose3(Rot3::Quaternion(q_w_curr.w(), q_w_curr.x(), q_w_curr.y(), q_w_curr.z()),
                                                   Point3(t_w_curr.x(), t_w_curr.y(), t_w_curr.z())), priorNoise));
        // initialEstimate의 데이터 형식은 Values이며, 실제로는 맵(map)과 유사합니다.
        // 여기에서 0에 해당하는 값을 Pose3로 설정하여 초기 추정값을 저장합니다.
        initialEstimate.insert(0, Pose3(Rot3::Quaternion(q_w_curr.w(), q_w_curr.x(), q_w_curr.y(), q_w_curr.z()),
                                                    Point3(t_w_curr.x(), t_w_curr.y(), t_w_curr.z())));
        q_w_curr_last = q_w_curr;
        t_w_curr_last = t_w_curr;
    }

    else{
    // 이전 키프레임이 존재하는 경우, 현재 로봇 위치와 이전 로봇 위치 사이의 변환을 계산합니다.
    gtsam::Pose3 poseFrom = Pose3(Rot3::Quaternion(q_w_curr_last.w(), q_w_curr_last.x(), q_w_curr_last.y(), q_w_curr_last.z()),
                                               Point3(t_w_curr_last.x(), t_w_curr_last.y(), t_w_curr_last.z()));
    gtsam::Pose3 poseTo   = Pose3(Rot3::Quaternion(q_w_curr.w(), q_w_curr.x(), q_w_curr.y(), q_w_curr.z()),
                                               Point3(t_w_curr.x(), t_w_curr.y(), t_w_curr.z()));
    
    // 이전 키프레임에서 현재 키프레임으로의 변환을 그래프에 추가합니다.
    gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->points.size()-1, cloudKeyPoses3D->points.size(), poseFrom.between(poseTo), odometryNoise));
    
    // initialEstimate에 현재 키프레임의 위치와 회전 정보를 추가합니다.
    initialEstimate.insert(cloudKeyPoses3D->points.size(), poseTo);
}

// 그래프 최적화를 업데이트하고 최적화를 수행합니다.
isam->update(gtSAMgraph, initialEstimate);
isam->update();

// 그래프 및 초기 추정값을 초기화합니다.
gtSAMgraph.resize(0);
initialEstimate.clear();

// 키프레임의 3D 위치 정보를 업데이트하고 저장합니다.
PointType thisPose3D;
PointTypePose thisPose6D;
Pose3 latestEstimate;

// 최신 추정값을 계산하여 3D 위치 정보를 업데이트합니다.
isamCurrentEstimate = isam->calculateEstimate();
latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);

thisPose3D.x = latestEstimate.translation().x();
thisPose3D.y = latestEstimate.translation().y();
thisPose3D.z = latestEstimate.translation().z();
thisPose3D.intensity = cloudKeyPoses3D->points.size();

// 3D 위치 정보를 키프레임 리스트에 추가합니다.
cloudKeyPoses3D->push_back(thisPose3D);


  // 6D 포인트 클라우드에 위치 및 회전 정보를 추가합니다.
thisPose6D.x = thisPose3D.x;
thisPose6D.y = thisPose3D.y;
thisPose6D.z = thisPose3D.z;
thisPose6D.intensity = thisPose3D.intensity;
thisPose6D.q_x = latestEstimate.rotation().toQuaternion().x();
thisPose6D.q_y = latestEstimate.rotation().toQuaternion().y();
thisPose6D.q_z = latestEstimate.rotation().toQuaternion().z();
thisPose6D.q_w = latestEstimate.rotation().toQuaternion().w();
thisPose6D.time = timeLaserOdometry;
cloudKeyPoses6D->push_back(thisPose6D);

// 키프레임 리스트에 3D 위치 및 회전 정보를 추가합니다.
if (cloudKeyPoses3D->points.size() > 1){
    q_w_curr.x() = latestEstimate.rotation().toQuaternion().x();
    q_w_curr.y() = latestEstimate.rotation().toQuaternion().y();
    q_w_curr.z() = latestEstimate.rotation().toQuaternion().z();
    q_w_curr.w() = latestEstimate.rotation().toQuaternion().w();
    t_w_curr.x() = latestEstimate.translation().x();
    t_w_curr.y() = latestEstimate.translation().y();
    t_w_curr.z() = latestEstimate.translation().z();

    // 이전 위치와 회전 정보를 업데이트합니다.
    q_w_curr_last = q_w_curr;
    t_w_curr_last = t_w_curr;
}

// 현재 키프레임을 NDTCloudKeyFrames 리스트에 추가합니다.
pcl::PointCloud<PointType>::Ptr thisNDTKeyFrame(new pcl::PointCloud<PointType>());
pcl::copyPointCloud(*laserCloudNDTLast, *thisNDTKeyFrame);
NDTCloudKeyFrames.push_back(thisNDTKeyFrame);

// transformUpdate();

}

void correctPoses(){
    // 루프 클로저가 닫혔을 때만 실행합니다.
    if (aLoopIsClosed == true){
        // 최근의 NDT 클라우드 키프레임 리스트를 비웁니다.
        recentNDTCloudKeyFrames.clear();

        // 현재 추정된 포즈의 수를 가져옵니다.
        int numPoses = isamCurrentEstimate.size();

        // 모든 포즈에 대해 위치 정보를 업데이트합니다.
        for (int i = 0; i < numPoses; ++i)
        {
            // 3D 위치 정보를 업데이트합니다.
            cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
            cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
            cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

            // 6D 위치 정보도 3D 정보와 동일하게 업데이트합니다.
            cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
            cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
            cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;

            // 6D 포즈에서 회전 정보를 업데이트합니다.
            cloudKeyPoses6D->points[i].q_x = isamCurrentEstimate.at<Pose3>(i).rotation().toQuaternion().x();
            cloudKeyPoses6D->points[i].q_y = isamCurrentEstimate.at<Pose3>(i).rotation().toQuaternion().y();
            cloudKeyPoses6D->points[i].q_z = isamCurrentEstimate.at<Pose3>(i).rotation().toQuaternion().z();
            cloudKeyPoses6D->points[i].q_w = isamCurrentEstimate.at<Pose3>(i).rotation().toQuaternion().w();
        }

        // 위치 및 회전 정보를 업데이트합니다.
        transformUpdate();

        // 루프 클로저 상태를 다시 열린 상태로 설정합니다.
        aLoopIsClosed = false;
    }
}

void clearCloud() {
    // NDT 클라우드를 지웁니다.
    laserCloudNDTFromMap->clear();
    laserCloudNDTFromMapDS->clear();

    // Aft Mapped odometry를 발행합니다.
    nav_msgs::Odometry odomAftMapped;
    odomAftMapped.header.frame_id = "/map";
    odomAftMapped.child_frame_id = "/aft_mapped";
    odomAftMapped.header.stamp = ros::Time(timeLaserOdometry);
    odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
    odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
    odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
    odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
    odomAftMapped.pose.pose.position.x = t_w_curr.x();
    odomAftMapped.pose.pose.position.y = t_w_curr.y();
    odomAftMapped.pose.pose.position.z = t_w_curr.z();
    pubOdomAftMappedHighFrec.publish(odomAftMapped);

    // TF 변환을 설정하고 발행합니다.
    static tf::TransformBroadcaster pose_broadcaster;
    tf::StampedTransform odom_trans;
    odom_trans.stamp_ = odomAftMapped.header.stamp;
    odom_trans.frame_id_ = "/map";
    odom_trans.child_frame_id_ = "/base_link";

    // 변환 행렬을 설정합니다.
    odom_trans.setRotation(tf::Quaternion(odomAftMapped.pose.pose.orientation.x, odomAftMapped.pose.pose.orientation.y, odomAftMapped.pose.pose.orientation.z, odomAftMapped.pose.pose.orientation.w));
    odom_trans.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, odomAftMapped.pose.pose.position.y, odomAftMapped.pose.pose.position.z));
    pose_broadcaster.sendTransform(odom_trans);

    // 레이저 포즈 정보를 발행합니다.
    geometry_msgs::PoseStamped laserPose;
    laserPose.header = odomAftMapped.header;
    laserPose.pose = odomAftMapped.pose.pose;
    laserPath.header.stamp = odomAftMapped.header.stamp;
    laserPath.poses.push_back(laserPose);
    laserPath.header.frame_id = "/map";
    pubLaserPath.publish(laserPath);
}

bool detectLoopClosure(){

    // 최근 NDT 키프레임 클라우드와 인근 히스토리 NDT 키프레임 클라우드를 초기화합니다.
    latestNDTKeyFrameCloud->clear();
    nearHistoryNDTKeyFrameCloud->clear();
    nearHistoryNDTKeyFrameCloudDS->clear();

    // 리소스 할당을 위한 뮤텍스를 사용하여 스레드 간 동기화합니다.
    std::lock_guard<std::mutex> lock(mtx);

    // 현재 로봇 위치 주변의 history 키프레임을 검색합니다.
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
    kdtreeHistoryKeyPoses->radiusSearch(currentRobotPosPoint, 15.0, pointSearchIndLoop, pointSearchSqDisLoop, 0);
	
    closestHistoryFrameID = -1; // 가장 가까운 히스토리 프레임의 ID를 초기화합니다.
    for (int i = 0; i < pointSearchIndLoop.size(); ++i){
        int id = pointSearchIndLoop[i]; // 현재 검색된 키프레임의 ID를 가져옵니다.
        // 현재 키프레임과 검색된 키프레임의 시간 차이가 30초 이상인 경우를 확인합니다.
        if (abs(cloudKeyPoses6D->points[id].time - timeLaserOdometry) > 30.0){
            closestHistoryFrameID = id; // 차이가 30초 이상인 키프레임의 ID를 저장합니다.
            break; // 찾았으므로 루프를 종료합니다.
        }
    }
    if (closestHistoryFrameID == -1){
        // 찾은 키프레임과 현재 시간의 차이가 30초 미만인 경우, 루프 클로저를 감지할 수 없습니다.
        return false; // 루프 클로저 감지 실패를 반환합니다.
    }

    // 현재까지의 클라우드 키포인트 개수에서 1을 뺀 값을 최신 프레임의 인덱스로 설정
latestFrameIDLoopCloure = cloudKeyPoses3D->points.size() - 1;

// 최신 NDT 키프레임 클라우드에 현재 프레임의 변환된 NDT 클라우드를 더함
*latestNDTKeyFrameCloud += *transformPointCloud(NDTCloudKeyFrames[latestFrameIDLoopCloure], &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);

// 새로운 PCL 포인트 클라우드 'hahaCloud' 생성
pcl::PointCloud<PointType>::Ptr hahaCloud(new pcl::PointCloud<PointType>());

// 현재 NDT 키프레임 클라우드의 포인트 개수 저장
int cloudSize = latestNDTKeyFrameCloud->points.size();

// 포인트 클라우드의 모든 포인트를 반복하면서 다음 조건을 확인
for (int i = 0; i < cloudSize; ++i) {
    // 포인트의 intensity 값이 0 이상인 경우만 'hahaCloud'에 추가
    if ((int)latestNDTKeyFrameCloud->points[i].intensity >= 0) {
        hahaCloud->push_back(latestNDTKeyFrameCloud->points[i]);
    }
}

// 'latestNDTKeyFrameCloud' 클라우드를 초기화
latestNDTKeyFrameCloud->clear();

// 'latestNDTKeyFrameCloud' 클라우드를 'hahaCloud'로 대체
*latestNDTKeyFrameCloud = *hahaCloud;

	// 'historyKeyframeSearchNum'은 'utility.h' 파일에서 정의되며, 이 값은 25로 설정되어 있음
	// 가장 가까운 히스토리 프레임에서 현재 프레임을 기준으로 앞뒤로 25개의 프레임을 처리하기 위한 루프
	for (int j = -25; j <= 25; ++j) {
	    // 'closestHistoryFrameID'와 'j'를 더한 값이 유효한 범위를 벗어나면 다음 반복으로 넘어감
	    if (closestHistoryFrameID + j < 0 || closestHistoryFrameID + j > latestFrameIDLoopCloure)
	        continue;
	
	    // 'closestHistoryFrameID + j' 위치의 NDT 클라우드와 포즈를 이용하여 변환 후 'nearHistoryNDTKeyFrameCloud'에 추가
	    *nearHistoryNDTKeyFrameCloud += *transformPointCloud(NDTCloudKeyFrames[closestHistoryFrameID + j], &cloudKeyPoses6D->points[closestHistoryFrameID + j]);
	}
	
	// 다운샘플링 필터를 사용하여 'nearHistoryNDTKeyFrameCloud' 클라우드를 줄임
	downSizeFilterHistoryKeyFrames.setInputCloud(nearHistoryNDTKeyFrameCloud);
	downSizeFilterHistoryKeyFrames.filter(*nearHistoryNDTKeyFrameCloudDS);
	
	// 주석 처리된 코드 블록: 퍼블리셔를 통해 히스토리 키프레임 클라우드를 전송하는 부분으로 보이며 현재 주석 처리되어 있음
	// if (pubHistoryKeyFrames.getNumSubscribers() != 0){
	//     sensor_msgs::PointCloud2 cloudMsgTemp;
	//     pcl::toROSMsg(*nearHistorySurfKeyFrameCloudDS, cloudMsgTemp);
	//     cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
	//     cloudMsgTemp.header.frame_id = "/camera_init";
	//     pubHistoryKeyFrames.publish(cloudMsgTemp);
	// }
	
	// 함수가 성공적으로 실행되었음을 나타내는 true 값을 반환
	return true;

}

void performLoopClosure() {
    // 클라우드 키포인트가 비어있으면 함수를 종료
    if (cloudKeyPoses3D->points.empty() == true)
        return;
    
    // 잠재적인 루프 플래그가 false인 경우
    if (potentialLoopFlag == false) {
        // 루프 클로저를 감지하면 잠재적 루프 플래그를 true로 설정
        if (detectLoopClosure() == true) {
            potentialLoopFlag = true;
            // timeSaveFirstCurrentScanForLoopClosure = timeLaserOdometry;
            // 잠재적 루프를 발견한 경우, 잠재적 루프 플래그를 true로 설정
        }

        // 잠재적 루프 플래그가 여전히 false인 경우 함수를 종료
        if (potentialLoopFlag == false)
            return;
    }

    // 잠재적 루프 클로저 검출 후, 루프 클로저 수행
    cout << "========================================================" << endl;
    potentialLoopFlag = false;

    // TODO: 나중에 NDT(정밀도 군집화 기술)로 변경할 수 있음
    // 아래 주석 처리된 코드는 현재 ICP(Iterative Closest Point) 알고리즘을 사용하고 있음
    // pcl::IterativeClosestPoint<PointType, PointType> icp;
    // icp.setMaxCorrespondenceDistance(100);
    // icp.setMaximumIterations(100);
    // icp.setTransformationEpsilon(1e-6);
    // icp.setEuclideanFitnessEpsilon(1e-6);
    // // RANSAC 실행 횟수 설정
    // icp.setRANSACIterations(0);
		
	// 아래 코드 블록은 ICP 알고리즘을 사용하여 포인트 클라우드를 정렬하는 부분이 주석 처리되어 있음
	// icp.setInputSource(latestNDTKeyFrameCloud);
	// // detectLoopClosure() 함수에서 다운샘플링된 nearHistorySurfKeyFrameCloudDS를 입력으로 사용
	// icp.setInputTarget(nearHistoryNDTKeyFrameCloudDS);
	// pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
	// // ICP를 사용하여 포인트 클라우드 정렬 수행
	// icp.align(*unused_result);
	
	// // 높은 매칭 점수일 때 직접 반환하는 이유는 높은 점수는 너무 많은 노이즈를 나타낼 수 있기 때문입니다.
	// if (icp.hasConverged() == false || icp.getFitnessScore() > 0.3)
	//     return; 
	
	// 아래 코드는 주석 처리된 부분과 관련하여 NDT(정규 분포 변환) 알고리즘을 사용하는 부분입니다.
	// pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>::Ptr ndt_loop(new pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>());
	
	// 최대 대응 거리를 설정합니다. (100)
	ndt_loop->setMaxCorrespondenceDistance(100);
	// 최대 반복 횟수를 설정합니다. (100)
	ndt_loop->setMaximumIterations(100);
	// 변환 엡실론(epsilon)을 설정합니다. (1e-6)
	ndt_loop->setTransformationEpsilon(1e-6);
	// 유클리드 거리 피트니스 엡실론(epsilon)을 설정합니다. (1e-6)
	ndt_loop->setEuclideanFitnessEpsilon(1e-6);
	// RANSAC 반복 횟수를 설정합니다. (0)
	ndt_loop->setRANSACIterations(0);
	
	// 해상도를 설정합니다. (5)
	ndt_loop->setResolution(5);
	// 스텝 사이즈를 설정합니다. (0.5)
	ndt_loop->setStepSize(0.5);
	// 이웃 검색 방법을 설정합니다. (pclomp::DIRECT1)
	ndt_loop->setNeighborhoodSearchMethod(pclomp::DIRECT1);
	// 사용할 스레드 수를 설정합니다. (4)
	ndt_loop->setNumThreads(4);
	// 입력 클라우드를 설정합니다. (latestNDTKeyFrameCloud)
	ndt_loop->setInputCloud(latestNDTKeyFrameCloud);
	// 대상 클라우드를 설정합니다. (nearHistoryNDTKeyFrameCloudDS)
	ndt_loop->setInputTarget(nearHistoryNDTKeyFrameCloudDS);
	// 결과를 저장할 포인트 클라우드를 생성합니다.
	pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
	// NDT 매칭을 실행합니다.
	ndt_loop->align(*unused_result);


    	// 현재 NDT 매칭의 적합도 점수를 출력합니다.
	std::cout << "fit score: " << ndt_loop->getFitnessScore() << std::endl;
	
	// NDT 매칭이 수렴하지 않거나 적합도 점수가 0.5보다 크면 함수를 종료합니다.
	if (ndt_loop->hasConverged() == false || ndt_loop->getFitnessScore() > 0.5)
	    return;
	
	// PCL 라이브러리에서 제공하는 IterativeClosestPoint(ICP) 알고리즘을 초기화합니다.
	pcl::IterativeClosestPoint<PointType, PointType> icp;
	
	// 최대 대응 거리를 설정합니다. (100)
	icp.setMaxCorrespondenceDistance(100);
	
	// 최대 반복 횟수를 설정합니다. (100)
	icp.setMaximumIterations(100);
	
	// 변환 엡실론(epsilon)을 설정합니다. (1e-6)
	icp.setTransformationEpsilon(1e-6);
	
	// 유클리드 거리 피트니스 엡실론(epsilon)을 설정합니다. (1e-6)
	icp.setEuclideanFitnessEpsilon(1e-6);
	
	// RANSAC 반복 횟수를 설정합니다. (0)
	icp.setRANSACIterations(0);

	
	// 포인트 클라우드들을 정렬합니다.
	icp.setInputSource(latestNDTKeyFrameCloud);      // 원본 포인트 클라우드 설정
	icp.setInputTarget(nearHistoryNDTKeyFrameCloudDS); // 대상 포인트 클라우드 설정
	
	// 결과를 저장할 포인트 클라우드를 생성합니다.
	pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
	
	// ICP (Iterative Closest Point) 알고리즘을 실행하여 포인트 클라우드를 정렬합니다.
	icp.align(*unused_result);
	
	// ICP 결과의 적합도 점수를 출력합니다.
	std::cout << "fit score: " << icp.getFitnessScore() << std::endl;
	
	// ICP가 수렴하지 않거나 적합도 점수가 0.5보다 크면 함수를 종료합니다.
	if (icp.hasConverged() == false || icp.getFitnessScore() > 0.5)
	    return;
	
  // // 다음은 포인트 클라우드 ICP 수렴 및 노이즈 범위 내에서 처리되는 경우입니다.
// if (pubIcpKeyFrames.getNumSubscribers() != 0){

    // 새로운 포인트 클라우드를 생성합니다.
    //pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());

    // icp.getFinalTransformation()의 반환값은 Eigen::Matrix<Scalar, 4, 4> 형식입니다.

    // 최종 변환 행렬을 사용하여 포인트 클라우드를 변환합니다.
    //pcl::transformPointCloud(*latestSurfKeyFrameCloud, *closed_cloud, icp.getFinalTransformation());

    // 변환된 클라우드를 ROS 메시지로 변환합니다.
    //sensor_msgs::PointCloud2 cloudMsgTemp;
    //pcl::toROSMsg(*closed_cloud, cloudMsgTemp);

    // ROS 메시지의 타임스탬프를 설정합니다.
    //cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);

    // ROS 메시지의 프레임 ID를 설정합니다.
    //cloudMsgTemp.header.frame_id = "/camera_init";

    // 변환된 클라우드를 발행합니다.
    //pubIcpKeyFrames.publish(cloudMsgTemp);
// }



 // 포즈 및 관련 변수 선언
float x, y, z, roll, pitch, yaw;
Eigen::Affine3f correctionFrame;
correctionFrame = icp.getFinalTransformation();
cout << "loop: " << correctionFrame.matrix() << endl;

// 변환 행렬에서 변환 및 회전 각도 정보를 추출합니다.
pcl::getTranslationAndEulerAngles(correctionFrame, x, y, z, roll, pitch, yaw);

// 클라우드 포인트에서 변환 행렬을 생성합니다.
Eigen::Affine3f tWrong = pclPointToAffine3fLidar(cloudKeyPoses6D->points[latestFrameIDLoopCloure]);

// 보정된 변환 행렬을 계산합니다.
Eigen::Affine3f tCorrect = correctionFrame * tWrong;

// 보정된 변환 행렬에서 변환 및 회전 각도 정보를 다시 추출합니다.
pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);

// gtsam을 사용하여 포즈를 생성합니다.
gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
gtsam::Pose3 poseTo = pclPointTogtsamPose3(cloudKeyPoses6D->points[closestHistoryFrameID]);

// 노이즈 모델을 설정합니다.
gtsam::Vector Vector6(6);
float noiseScore = icp.getFitnessScore();
Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
constraintNoise = noiseModel::Diagonal::Variances(Vector6);

// 그래프에 제약 조건을 추가합니다.
std::lock_guard<std::mutex> lock(mtx);
gtSAMgraph.add(BetweenFactor<Pose3>(latestFrameIDLoopCloure, closestHistoryFrameID, poseFrom.between(poseTo), constraintNoise));

// ISAM을 업데이트하고 그래프를 재설정합니다.
isam->update(gtSAMgraph);
isam->update();
gtSAMgraph.resize(0);

// 루프가 닫혔음을 표시합니다.
aLoopIsClosed = true;
}

void run() {

    while(ros::ok()) {

        // 만약 새로운 Laser Cloud와 Odometry 메시지가 도착한 경우
        if(newLaserCloudNDTLast && newLaserOdometry) {
            newLaserCloudNDTLast = false; newLaserOdometry = false;
            std::lock_guard<std::mutex> lock(mtx);

            // 지난 처리 시간과의 시간 간격이 0.3초보다 큰 경우 (선택적인 조건)
            // if(timeLaserCloudNDTLast - timeLastProcessing >= 0.3) {
                timeLastProcessing = timeLaserCloudNDTLast;
                
                // 현재 포즈를 맵에 대한 포즈로 변환합니다.
                transformAssociateToMap();

                // 주변 키프레임을 추출합니다.
                extractSurroundingKeyFrames();

                // 현재 스캔을 다운샘플링합니다. (주석 처리된 부분)
                // downsampleCurrentScan();

                // NDT(정찰기반 동적 공간 분할) 등록을 수행합니다.
                ndtRegistration();

                // 키프레임과 팩터를 저장합니다.
                saveKeyFramesAndFactor();

                // TF를 발행합니다. (주석 처리된 부분)
                // publishTF();

                // 키포즈와 프레임을 발행합니다. (주석 처리된 부분)
                // publishKeyPosesAndFrames();

                // 클라우드를 초기화합니다.
                clearCloud();

                // 포즈를 보정합니다.
                correctPoses();

                auto t2 = ros::WallTime::now();
                // cout << "lidar mapping: " << (t2 - t1).toSec() * 1000 << "  ms" << endl;
            // }
        }            
    }
}

void loopClosureThread() {
    // 루프 클로저 기능이 비활성화된 경우 함수 종료
    if(loopClosureEnableFlag == false) 
        return;

    // 루프 클로저 스레드를 실행하는 무한 루프
    while(ros::ok()) {
        // 루프 클로저 동작 수행
        performLoopClosure();

        // 디버깅 목적으로 "sss"를 출력합니다.
        cout << "sss" << endl;

        // 스레드를 1초 동안 대기시킵니다.
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

void publishGlobalMap() {
    // 만약 구독자가 없으면 함수 종료
    if (pubLaserCloudSurround.getNumSubscribers() == 0)
        return;

    // 만약 키포즈 클라우드가 비어있으면 함수 종료
    if (cloudKeyPoses3D->points.empty() == true)
        return;

    // 주변에 시각화할 키프레임을 찾기 위해 KD 트리를 사용합니다.
    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;

    // 주변에 시각화할 키프레임을 찾기 위한 반경 검색을 수행합니다.
    mtx.lock();
    kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
    kdtreeGlobalMap->radiusSearch(currentRobotPosPoint, 500.0, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
    mtx.unlock();

    // 검색된 키포즈를 globalMapKeyPoses에 추가합니다.
    for (int i = 0; i < pointSearchIndGlobalMap.size(); ++i)
        globalMapKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);

    // 검색된 키포즈를 다운샘플링합니다.
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

    // 시각화 및 다운샘플링된 키포즈를 이용하여 키프레임을 추출합니다.
    for (int i = 0; i < globalMapKeyPosesDS->points.size(); ++i) {
        int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
        *globalMapKeyFrames += *transformPointCloud(NDTCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
    }

    // 추출된 키프레임을 다운샘플링합니다.
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);

    // ROS 메시지로 변환하여 발행합니다.
    sensor_msgs::PointCloud2 cloudMsgTemp;
    pcl::toROSMsg(*globalMapKeyFramesDS, cloudMsgTemp);
    cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
    cloudMsgTemp.header.frame_id = "/map";
    pubLaserCloudSurround.publish(cloudMsgTemp);

    // 사용한 데이터를 클리어합니다.
    globalMapKeyPoses->clear();
    globalMapKeyPosesDS->clear();
    globalMapKeyFrames->clear();
    globalMapKeyFramesDS->clear();
}


void visualizeGlobalMapThread() {
    // 시각화 속도를 설정합니다. (0.2 Hz로 설정)
    ros::Rate rate(0.2);

    while (ros::ok()) {
        // 전역 지도를 발행하는 함수를 호출합니다.
        publishGlobalMap();

        // 시각화 속도에 따라 스레드를 대기시킵니다.
        rate.sleep();
    }
	
// Eigen::Affine3f transform = Eigen::Affine3f::Identity();
// 변환 행렬을 단위 행렬로 초기화합니다.

// transform.rotate(Eigen::AngleAxisf(M_PI/2, Eigen::Vector3f(1,0,0)));
// 변환 행렬에 X축 주위로 90도 회전을 적용합니다.

// pcl::transformPointCloud(*globalMapKeyFramesDS, *globalMapKeyFramesDS, transform);
// 포인트 클라우드에 위에서 정의한 변환을 적용합니다.

// Eigen::Affine3f transform1 = Eigen::Affine3f::Identity();
// 또 다른 변환 행렬을 단위 행렬로 초기화합니다.

// transform1.rotate(Eigen::AngleAxisf(M_PI/2, Eigen::Vector3f(0,0,1)));
// 두 번째 변환 행렬에 Z축 주위로 90도 회전을 적용합니다.

// pcl::transformPointCloud(*globalMapKeyFramesDS, *globalMapKeyFramesDS, transform1);
// 포인트 클라우드에 두 번째 변환을 적용합니다.

// // save final point cloud
// // 최종 포인트 클라우드를 저장합니다.
// pcl::io::savePCDFileASCII(fileDirectory+"finalCloud.pcd", *globalMapKeyFramesDS);
// 파일로 저장하고 파일 이름은 "finalCloud.pcd"로 합니다.

// string cornerMapString = "cornerMap.pcd";
// 포인트 클라우드 파일 이름을 "cornerMap.pcd"로 설정합니다.

// string surfaceMapString = "surfaceMap.pcd";
// 포인트 클라우드 파일 이름을 "surfaceMap.pcd"로 설정합니다.

// string trajectoryString = "/tmp/trajectory.pcd";
// 포인트 클라우드 파일 이름을 "/tmp/trajectory.pcd"로 설정합니다.

// pcl::PointCloud<PointType>::Ptr cornerMapCloud(new pcl::PointCloud<PointType>());
// PointType 형식의 포인트 클라우드 포인터를 cornerMapCloud라는 변수로 생성합니다.

// pcl::PointCloud<PointType>::Ptr cornerMapCloudDS(new pcl::PointCloud<PointType>());
// PointType 형식의 포인트 클라우드 포인터를 cornerMapCloudDS라는 변수로 생성합니다.

// pcl::PointCloud<PointType>::Ptr surfaceMapCloud(new pcl::PointCloud<PointType>());
// PointType 형식의 포인트 클라우드 포인터를 surfaceMapCloud라는 변수로 생성합니다.

// pcl::PointCloud<PointType>::Ptr surfaceMapCloudDS(new pcl::PointCloud<PointType>());
// PointType 형식의 포인트 클라우드 포인터를 surfaceMapCloudDS라는 변수로 생성합니다.

// for(int i = 0; i < cornerCloudKeyFrames.size(); i++) {
// cornerCloudKeyFrames 배열의 크기만큼 반복합니다.

// *cornerMapCloud  += *transformPointCloud(cornerCloudKeyFrames[i],   &cloudKeyPoses6D->points[i]);
// cornerCloudKeyFrames[i]를 cloudKeyPoses6D->points[i]로 변환한 포인트 클라우드를 cornerMapCloud에 추가합니다.

// *surfaceMapCloud += *transformPointCloud(surfCloudKeyFrames[i],     &cloudKeyPoses6D->points[i]);
// surfCloudKeyFrames[i]를 cloudKeyPoses6D->points[i]로 변환한 포인트 클라우드를 surfaceMapCloud에 추가합니다.

// *surfaceMapCloud += *transformPointCloud(outlierCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
// outlierCloudKeyFrames[i]를 cloudKeyPoses6D->points[i]로 변환한 포인트 클라우드를 surfaceMapCloud에 추가합니다.
// }

    // downSizeFilterCorner.setInputCloud(cornerMapCloud);
// downSizeFilterCorner에 cornerMapCloud를 입력으로 설정합니다.

// downSizeFilterCorner.filter(*cornerMapCloudDS);
// downSizeFilterCorner를 사용하여 cornerMapCloud를 다운샘플링하고 cornerMapCloudDS에 저장합니다.

// downSizeFilterSurf.setInputCloud(surfaceMapCloud);
// downSizeFilterSurf에 surfaceMapCloud를 입력으로 설정합니다.

// downSizeFilterSurf.filter(*surfaceMapCloudDS);
// downSizeFilterSurf를 사용하여 surfaceMapCloud를 다운샘플링하고 surfaceMapCloudDS에 저장합니다.

// Eigen::Affine3f transform2 = Eigen::Affine3f::Identity();
// 변환 행렬을 단위 행렬로 초기화합니다.

// transform2.rotate(Eigen::AngleAxisf(M_PI/2, Eigen::Vector3f(1,0,0)));
// 변환 행렬에 X축 주위로 90도 회전을 적용합니다.

// pcl::transformPointCloud(*cornerMapCloud, *cornerMapCloud, transform2);
// cornerMapCloud에 위에서 정의한 변환을 적용합니다.

// Eigen::Affine3f transform3 = Eigen::Affine3f::Identity();
// 또 다른 변환 행렬을 단위 행렬로 초기화합니다.

// transform3.rotate(Eigen::AngleAxisf(M_PI/2, Eigen::Vector3f(0,0,1)));
// 두 번째 변환 행렬에 Z축 주위로 90도 회전을 적용합니다.

// pcl::transformPointCloud(*cornerMapCloud, *cornerMapCloud, transform3);
// cornerMapCloud에 두 번째 변환을 적용합니다.

// Eigen::Affine3f transform4 = Eigen::Affine3f::Identity();
// 다른 변환 행렬을 단위 행렬로 초기화합니다.

// transform4.rotate(Eigen::AngleAxisf(M_PI/2, Eigen::Vector3f(1,0,0)));
// 변환 행렬에 X축 주위로 90도 회전을 적용합니다.

// pcl::transformPointCloud(*surfaceMapCloud, *surfaceMapCloud, transform4);
// surfaceMapCloud에 위에서 정의한 변환을 적용합니다.

// Eigen::Affine3f transform5 = Eigen::Affine3f::Identity();
// 또 다른 변환 행렬을 단위 행렬로 초기화합니다.

// transform5.rotate(Eigen::AngleAxisf(M_PI/2, Eigen::Vector3f(0,0,1)));
// 다섯 번째 변환 행렬에 Z축 주위로 90도 회전을 적용합니다.

// pcl::transformPointCloud(*surfaceMapCloud, *surfaceMapCloud, transform5);
// surfaceMapCloud에 다섯 번째 변환을 적용합니다.

// pcl::io::savePCDFileASCII(fileDirectory+"cornerMap.pcd", *cornerMapCloud);
// cornerMapCloud를 "cornerMap.pcd" 파일로 저장합니다.

// pcl::io::savePCDFileASCII(fileDirectory+"surfaceMap.pcd", *surfaceMapCloud);
// surfaceMapCloud를 "surfaceMap.pcd" 파일로 저장합니다.

// // pcl::io::savePCDFileASCII(fileDirectory+"trajectory.pcd", *cloudKeyPoses3D);
// 주석 처리된 부분은 cloudKeyPoses3D를 파일로 저장하는 부분으로 보이며, 필요에 따라 주석을 해제하여 사용할 수 있습니다.
}




int main(int argc, char** argv) {
    // ROS 초기화 및 노드 이름 설정
    ros::init(argc, argv, "test_mapping_node");
    ros::NodeHandle node = ros::NodeHandle();

     // gtsam 라이브러리를 사용한 그래프 최적화를 위한 매개변수 설정
    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01; // 다시 선형화할 임계값 설정
    parameters.relinearizeSkip = 1; // 재선형화를 건너뛸 간격 설정
    isam = new ISAM2(parameters); // ISAM2 객체 생성


    // 라이다 포인트 클라우드 데이터를 수신하기 위한 ROS 구독자 설정
    ros::Subscriber subNDTCloud = node.subscribe("/laser_cloud_NDT", 100, &ndtCloudCallback);
    ros::Subscriber subLaserOdometry_Loam = node.subscribe("/laser_odom_loam", 100, &laserOdometryLoamCallback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber subLaserOdometry_ndt = node.subscribe("/laser_odom_ndt", 100, &laserOdometryNDTCallback, ros::TransportHints().tcpNoDelay());

    // 초기화된 위치로 매핑된 로봇의 고주파 오도메트리를 발행하기 위한 ROS 발행자 설정
    pubOdomAftMappedHighFrec = node.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 10);

    // 주변 라이다 클라우드 데이터를 발행하기 위한 ROS 발행자 설정
    pubLaserCloudSurround = node.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 2);

    // 로봇의 매핑된 경로를 발행하기 위한 ROS 발행자 설정
    pubLaserPath = node.advertise<nav_msgs::Path>("/laser_after_map_path", 10);

	// 최근 NDT 키 프레임 라이다 클라우드 초기화
	laserCloudNDTLast.reset(new pcl::PointCloud<PointType>());
	// 최신 NDT 키 프레임 클라우드 초기화
	latestNDTKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
	// 맵에서 가져온 NDT 라이다 클라우드 초기화
	laserCloudNDTFromMap.reset(new pcl::PointCloud<PointType>());
	// 다운샘플링된 맵에서 가져온 NDT 라이다 클라우드 초기화
	laserCloudNDTFromMapDS.reset(new pcl::PointCloud<PointType>());
	// 근접한 과거 NDT 키 프레임 클라우드 초기화
	nearHistoryNDTKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
	// 다운샘플링된 근접한 과거 NDT 키 프레임 클라우드 초기화
	nearHistoryNDTKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());
	// 포인트 클라우드의 키 포즈 3D 초기화
	cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
	// 포인트 클라우드의 키 포즈 6D 초기화
	cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
	// 주변 키 포즈 초기화
	surroundingKeyPoses.reset(new pcl::PointCloud<PointType>());
	// 다운샘플링된 주변 키 포즈 초기화
	surroundingKeyPosesDS.reset(new pcl::PointCloud<PointType>());
	
	
	// 전역 맵을 위한 KD 트리 초기화
	kdtreeGlobalMap.reset(new pcl::KdTreeFLANN<PointType>());
	// 전역 맵 키 포즈 초기화
	globalMapKeyPoses.reset(new pcl::PointCloud<PointType>());
	// 다운샘플링된 전역 맵 키 포즈 초기화
	globalMapKeyPosesDS.reset(new pcl::PointCloud<PointType>());
	// 전역 맵 키 프레임 초기화
	globalMapKeyFrames.reset(new pcl::PointCloud<PointType>());
	// 다운샘플링된 전역 맵 키 프레임 초기화
	globalMapKeyFramesDS.reset(new pcl::PointCloud<PointType>());

	downSizeFilterNDT.setLeafSize(0.2, 0.2, 0.2); // NDT 필터의 리프 크기 설정
	downSizeFilterHistoryKeyFrames.setLeafSize(0.2, 0.2, 0.2); // 역사 키프레임 필터의 리프 크기 설정
	downSizeFilterSurroundingKeyPoses.setLeafSize(0.4, 0.4, 0.4); // 주변 키포즈 필터의 리프 크기 설정
	downSizeFilterGlobalMapKeyPoses.setLeafSize(1.0, 1.0, 1.0); // 전역 맵 시각화를 위한 전역 맵 키포즈 필터의 리프 크기 설정
	downSizeFilterGlobalMapKeyFrames.setLeafSize(0.4, 0.4, 0.4); // 전역 맵 시각화를 위한 전역 맵 키프레임 필터의 리프 크기 설정
	
	kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>()); // 주변 키포즈를 위한 KD 트리 초기화
	kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>()); // 역사 키포즈를 위한 KD 트리 초기화
	
	gtsam::Vector Vector6(6); // 6차원 벡터 생성
	Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6; // 6차원 벡터에 값을 할당
	priorNoise = noiseModel::Diagonal::Variances(Vector6); // 사전 노이즈 모델 설정
	odometryNoise = noiseModel::Diagonal::Variances(Vector6); // 오도메트리 노이즈 모델 설정
	
	std::thread looprun(&run); // run 함수를 실행하는 쓰레드 생성
	std::thread loopthread(&loopClosureThread); // loopClosureThread 함수를 실행하는 쓰레드 생성
	std::thread visualizeMapThread(&visualizeGlobalMapThread); // visualizeGlobalMapThread 함수를 실행하는 쓰레드 생성

    

	pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>::Ptr ndt(new pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>());
	// 포인트 클라우드를 위한 정규 분포 변환 객체(ndt)를 생성합니다.
	
	ndt->setMaxCorrespondenceDistance(50);
	// 최대 일치 거리(Max Correspondence Distance)를 50으로 설정합니다.
	
	ndt->setTransformationEpsilon(0.0001);
	// 변환 엡실론(Transformation Epsilon)을 0.0001로 설정합니다.
	
	ndt->setEuclideanFitnessEpsilon(0.0001);
	// 유클리드 피트니스 엡실론(Euclidean Fitness Epsilon)을 0.0001로 설정합니다.
	
	ndt->setResolution(2.0);
	// 해상도(Resolution)를 2.0으로 설정합니다.
	
	ndt->setStepSize(0.05);
	// 스텝 크기(Step Size)를 0.05로 설정합니다.
	
	ndt->setMaximumIterations(100);
	// 최대 반복 횟수(Maximum Iterations)를 100으로 설정합니다.
	
	ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
	// 이웃 검색 방법(Neighborhood Search Method)을 DIRECT1로 설정합니다.
	
	ndt->setNumThreads(4);
	// 병렬 처리를 위한 쓰레드 수(Number of Threads)를 4로 설정합니다.
	
	registration = ndt;
	// 등록 객체(registration)를 위에서 설정한 ndt로 초기화합니다.
	
	ros::Rate rate(100);
	// ROS 시간 주기(rate)를 100Hz로 설정합니다.


    
while(ros::ok()) {
    // ROS가 정상 동작 중인 동안 반복합니다.

    // run();
    // run 함수를 호출하는 주석 처리된 코드입니다. 나중에 활성화할 수 있습니다.
    ros::spinOnce();
    // ROS 노드의 콜백 함수를 실행하여 메시지 처리를 합니다.

    if(newLaserOdometry_loam)
    {
        // newLaserOdometry_loam이 true인 경우에만 실행합니다.

        Eigen::Quaterniond q_tmp(1, 0, 0, 0);
        Eigen::Vector3d t_tmp(0, 0, 0);

        // if(odometry_weight > 0.5) {
        // odometry_weight가 0.5보다 큰 경우의 조건문입니다. 주석 처리되어 있습니다.

        q_tmp = q_wodom_curr_loam;
        t_tmp = t_wodom_curr_loam;
        // q_tmp와 t_tmp를 q_wodom_curr_loam과 t_wodom_curr_loam 값으로 설정합니다.

        // } 
        // else
        // {
        //     double w = (0.5 - odometry_weight) * 0.4;

        //     q_tmp.x() = w * q_wodom_curr_ndt.x() + (1 - w) * q_wodom_curr_loam.x();
        //     q_tmp.y() = w * q_wodom_curr_ndt.y() + (1 - w) * q_wodom_curr_loam.y();
        //     q_tmp.z() = w * q_wodom_curr_ndt.z() + (1 - w) * q_wodom_curr_loam.z();
        //     q_tmp.w() = w * q_wodom_curr_ndt.w() + (1 - w) * q_wodom_curr_loam.w();
        //     q_tmp.normalize();

        //     t_tmp = w * t_wodom_curr_ndt + (1 - w) * t_wodom_curr_loam;
        // }

        t_wodom_curr = t_wodom_curr + q_wodom_curr * t_tmp;
        q_wodom_curr = q_wodom_curr * q_tmp;
        // t_wodom_curr와 q_wodom_curr을 갱신합니다.

        // run();
        // run 함수를 호출하는 주석 처리된 코드입니다. 나중에 활성화할 수 있습니다.

        // high frequence publish
        // 높은 주파수로 발행하는 부분의 주석 처리된 코드입니다.

        // Eigen::Quaterniond q_w_curr_tmp = q_wmap_wodom * q_wodom_curr;
        // Eigen::Vector3d t_w_curr_tmp = q_wmap_wodom * t_wodom_curr + t_wmap_wodom; 

        // nav_msgs::Odometry odomAftMapped;
        // odomAftMapped.header.frame_id = "/map";
        // odomAftMapped.child_frame_id = "/aft_mapped";
        // odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        // odomAftMapped.pose.pose.orientation.x = q_w_curr_tmp.x();
        // odomAftMapped.pose.pose.orientation.y = q_w_curr_tmp.y();
        // odomAftMapped.pose.pose.orientation.z = q_w_curr_tmp.z();
        // odomAftMapped.pose.pose.orientation.w = q_w_curr_tmp.w();
        // odomAftMapped.pose.pose.position.x = t_w_curr_tmp.x();
        // odomAftMapped.pose.pose.position.y = t_w_curr_tmp.y();
        // odomAftMapped.pose.pose.position.z = t_w_curr_tmp.z();
        // pubOdomAftMappedHighFrec.publish(odomAftMapped);

        // geometry_msgs::PoseStamped laserPose;
        // laserPose.header = odomAftMapped.header;
        // laserPose.pose = odomAftMapped.pose.pose;
        // laserPath.header.stamp = odomAftMapped.header.stamp;
        // laserPath.poses.push_back(laserPose);
        // laserPath.header.frame_id = "/map";
        // pubLaserPath.publish(laserPath);

        newLaserOdometry_loam = false;
        newLaserOdometry_ndt = false;
        newLaserOdometry = true;
        // newLaserOdometry_loam 및 다른 변수들을 초기화합니다.
    }
    
    rate.sleep();
    // 루프를 주어진 속도(rate)에 따라 대기합니다.
}
looprun.join();
loopthread.join();
visualizeMapThread.join();

return 0;
}

