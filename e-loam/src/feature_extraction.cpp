// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#include <cmath>
#include <vector>
#include <string>
// C++ 헤더 파일 포함: 수학 함수, 벡터, 문자열 등을 다루기 위해 사용됩니다.

// 아래 두 줄은 주석 처리되어 있습니다.
// #include "aloam_velodyne/common.h"
// #include "aloam_velodyne/tic_toc.h"
// 이 코드에서는 이 헤더 파일들이 필요하지 않은 것으로 보입니다.

// 주석 처리된 헤더 파일: 로봇의 위치 및 이동 정보를 다루는 헤더 파일로 보입니다.
// #include <nav_msgs/Odometry.h>

#include <opencv/cv.h>
// OpenCV 라이브러리의 헤더 파일을 포함하고 있습니다. 이미지 처리 및 컴퓨터 비전에 사용됩니다.

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
// PCL (Point Cloud Library) 라이브러리의 헤더 파일들을 포함하고 있습니다.
// PCL은 3D 포인트 클라우드 데이터를 다루는데 사용됩니다.

#include <ros/ros.h>
// ROS (로봇 운영 체제) 라이브러리의 헤더 파일을 포함하고 있습니다.

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
// ROS에서 사용되는 센서 데이터 메시지 유형의 헤더 파일들을 포함하고 있습니다.

#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
// ROS의 변환 (transform) 관련 헤더 파일들을 포함하고 있습니다.

using namespace std;
// C++ 표준 라이브러리(std)의 네임스페이스를 사용하겠다는 선언입니다.

typedef pcl::PointXYZI PointType;
// pcl(Point Cloud Library)의 PointXYZI 형식을 사용하기 위한 정의입니다.

struct PointXYZIRT
{
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;
  uint16_t ring;
  float time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
// PointXYZIRT라는 사용자 정의 데이터 구조를 정의합니다.
// PCL_ADD_POINT4D 및 PCL_ADD_INTENSITY 매크로를 사용하여 구조체에 포인트와 강도를 추가합니다.
// ring 및 time 멤버 변수도 있습니다.

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRT,  
  (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
  (uint16_t, ring, ring) (float, time, time)
)
// PCL에서 사용자 정의 포인트 구조체(PointXYZIRT)를 등록하는 매크로입니다.
// 각 변수와 그들의 이름, 유형이 등록됩니다.

using std::atan2;
using std::cos;
using std::sin;
// C++ 표준 라이브러리의 atan2, cos, sin 함수를 사용하겠다는 선언입니다.


// const double scanPeriod = 0.1;
// 스캔 주기를 나타내는 상수입니다. 주석 처리되어 있으므로 현재 사용되지 않는 것으로 보입니다.

const int systemDelay = 0;
// 시스템 지연 시간을 나타내는 정수 상수입니다.

int systemInitCount = 0;
// 시스템 초기화 카운트를 나타내는 변수입니다.

bool systemInited = false;
// 시스템이 초기화되었는지 여부를 나타내는 불리언 변수입니다.

int N_SCANS = 0;
// 스캔 수를 나타내는 정수 변수입니다.

float cloudCurvature[400000];
// 포인트 클라우드의 곡률 정보를 저장하는 배열입니다.

float cloudIntensityDiff[400000];
// 포인트 클라우드의 강도 차이 정보를 저장하는 배열입니다.

int cloudSortInd[400000];
// 포인트 클라우드를 정렬하는 데 사용되는 인덱스 배열입니다.

int cloudNeighborPicked[400000];
// 선택된 포인트 이웃 정보를 저장하는 배열입니다.

int cloudLabel[400000];
// 포인트 클라우드의 레이블 정보를 저장하는 배열입니다.

bool comp_corner (int i, int j) { return (cloudCurvature[i] < cloudCurvature[j]); }
// 두 포인트의 곡률 정보를 비교하는 비교 함수입니다.

bool comp_intensity (int i, int j) { return (cloudIntensityDiff[i] < cloudIntensityDiff[j]); }
// 두 포인트의 강도 차이 정보를 비교하는 비교 함수입니다.

pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreePlanarCloud(new pcl::KdTreeFLANN<pcl::PointXYZI>());
// PCL(Point Cloud Library)에서 사용되는 KdTreeFLANN 객체를 생성하고 포인터로 가리키는 변수를 초기화합니다.

ros::Publisher pubCornerPointsSharp;
// ROS에서 사용할 "pubCornerPointsSharp"라는 이름의 Publisher(발행자) 객체를 정의합니다.
// 이 Publisher는 날카로운 모서리 포인트 데이터를 발행할 때 사용됩니다.

ros::Publisher pubCornerPointsLessSharp;
// ROS에서 사용할 "pubCornerPointsLessSharp"라는 이름의 Publisher 객체를 정의합니다.
// 이 Publisher는 날카로운 모서리 포인트 중 일부 데이터를 발행할 때 사용됩니다.

ros::Publisher pubSurfPointsFlat;
// ROS에서 사용할 "pubSurfPointsFlat"라는 이름의 Publisher 객체를 정의합니다.
// 이 Publisher는 평탄한 표면 포인트 데이터를 발행할 때 사용됩니다.

ros::Publisher pubSurfPointsLessFlat;
// ROS에서 사용할 "pubSurfPointsLessFlat"라는 이름의 Publisher 객체를 정의합니다.
// 이 Publisher는 평탄한 표면 포인트 중 일부 데이터를 발행할 때 사용됩니다.

ros::Publisher pubTexturePointSharp;
// ROS에서 사용할 "pubTexturePointSharp"라는 이름의 Publisher 객체를 정의합니다.
// 이 Publisher는 질감(텍스처) 정보가 있는 날카로운 모서리 포인트 데이터를 발행할 때 사용됩니다.

ros::Publisher pubTexturePointLessSharp;
// ROS에서 사용할 "pubTexturePointLessSharp"라는 이름의 Publisher 객체를 정의합니다.
// 이 Publisher는 질감 정보가 있는 날카로운 모서리 포인트 중 일부 데이터를 발행할 때 사용됩니다.

ros::Publisher PubNDTPoints;
// ROS에서 사용할 "PubNDTPoints"라는 이름의 Publisher 객체를 정의합니다.
// 이 Publisher는 Normal Distributions Transform(NDT) 포인트 데이터를 발행할 때 사용됩니다.

double MINIMUM_RANGE = 0.1;
// 최소 거리를 나타내는 실수형 상수를 정의합니다. 이 값은 0.1로 설정되어 있습니다.

template <typename PointT>
void RemoveNaNFromPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
                              pcl::PointCloud<PointT> &cloud_out,
                              std::vector<int> &index)
{
  // 입력 클라우드와 출력 클라우드가 다를 경우, 출력 클라우드를 준비합니다.
  if (&cloud_in != &cloud_out)
  {
    cloud_out.header = cloud_in.header;
    cloud_out.points.resize (cloud_in.points.size ());
  }
  
  // 인덱스 벡터의 크기를 입력 클라우드와 동일하게 설정합니다.
  index.resize (cloud_in.points.size ());
  size_t j = 0;

  // 입력 클라우드가 밀집 데이터인 경우, NaN 값을 확인할 필요가 없습니다.
  if (cloud_in.is_dense)
  {
    // 데이터를 단순히 복사합니다.
    cloud_out = cloud_in;
    for (j = 0; j < cloud_out.points.size (); ++j)
      index[j] = static_cast<int>(j);
  }
  else
{
  for (size_t i = 0; i < cloud_in.points.size (); ++i)
  {
    // NaN이 아닌 포인트를 선택하여 출력 클라우드와 인덱스를 업데이트합니다.
    if (!pcl_isfinite (cloud_in.points[i].x) || 
        !pcl_isfinite (cloud_in.points[i].y) || 
        !pcl_isfinite (cloud_in.points[i].z) ||
        !pcl_isfinite (cloud_in.points[i].intensity) ||
        !pcl_isfinite (cloud_in.points[i].ring))
      continue;
    cloud_out.points[j] = cloud_in.points[i];
    index[j] = static_cast<int>(i);
    j++;
  }

  // 포인트 클라우드와 인덱스가 업데이트되었을 경우
  if (j != cloud_in.points.size ())
  {
    // 올바른 크기로 재조정
    cloud_out.points.resize (j);
    index.resize (j);
  }

  // 출력 클라우드의 높이와 너비 설정
  cloud_out.height = 1;
  cloud_out.width  = static_cast<uint32_t>(j);

  // 나쁜 포인트를 제거했으므로 클라우드는 밀집된(dense) 상태임을 표시합니다.
  // 주의: 'dense'는 '정돈된(organized)'이 아님에 유의하세요.
  cloud_out.is_dense = true;
}

template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                            pcl::PointCloud<PointT> &cloud_out, float thres)
{
    // 입력 클라우드와 출력 클라우드가 다를 경우, 출력 클라우드를 준비합니다.
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    // 입력 클라우드의 모든 포인트를 확인합니다.
    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        // 거리 임계값(thres)보다 작으면 해당 포인트를 건너뜁니다.
        if (cloud_in.points[i].x * cloud_in.points[i].x + 
            cloud_in.points[i].y * cloud_in.points[i].y + 
            cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;

        // 거리 임계값(thres)보다 크면 너무 멀리 있는 것으로 판단하고 건너뜁니다.
        if (cloud_in.points[i].x * cloud_in.points[i].x + 
            cloud_in.points[i].y * cloud_in.points[i].y + 
            cloud_in.points[i].z * cloud_in.points[i].z > 10000)
            continue;

        // 조건을 만족하는 포인트를 출력 클라우드에 저장하고 인덱스를 증가시킵니다.
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }

    // 출력 클라우드의 크기를 업데이트합니다.
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    // 출력 클라우드의 높이와 너비를 설정하고 클라우드는 밀집된(dense) 상태임을 표시합니다.
    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}


void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    if (!systemInited)
    {
        // 시스템이 초기화되지 않았다면
        systemInitCount++;
        
        // 초기화 지연 시간(systemDelay) 이상 지난 경우
        if (systemInitCount >= systemDelay)
        {
            // 시스템이 초기화되었음을 표시합니다.
            systemInited = true;
        }
        else
        {
            // 초기화 지연 중이므로 더 이상 처리하지 않고 함수를 종료합니다.
            return;
        }

    auto t1 = ros::WallTime::now();
    // 현재 시간을 측정하기 위한 변수 t1을 생성하고 초기화합니다.
    
    std::vector<int> scanStartInd(N_SCANS, 0);
    // N_SCANS 개수만큼의 요소를 가지는 정수 벡터 scanStartInd를 생성하고 초기화합니다.
    
    std::vector<int> scanEndInd(N_SCANS, 0);
    // N_SCANS 개수만큼의 요소를 가지는 정수 벡터 scanEndInd를 생성하고 초기화합니다.
    
    pcl::PointCloud<PointXYZIRT> laserCloudIn;
    // PCL(Point Cloud Library)의 PointXYZIRT 유형을 가진 포인트 클라우드 laserCloudIn을 생성합니다.
    
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
    // ROS 메시지를 PCL 포인트 클라우드로 변환하여 laserCloudIn에 저장합니다.
    
    std::vector<int> indices;
    // 정수 벡터 indices를 생성합니다.
    
    RemoveNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    // laserCloudIn 포인트 클라우드에서 NaN 값을 제거하고 결과를 laserCloudIn에 다시 저장합니다.
    // 제거된 포인트의 인덱스 정보는 indices 벡터에 저장됩니다.
    
    if(laserCloudIn.is_dense == false) 
    {
        // 포인트 클라우드가 밀집 형식이 아닌 경우 오류 메시지를 출력하고 함수를 종료합니다.
        cout << "Point cloud is not in dense format, please remove NaN points first!" << endl;
        return;
    }

    static int ringFlag = 0;
    // 정적 정수 변수 ringFlag를 생성하고 초기화합니다.
    
    if(ringFlag == 0) 
    {
        // ringFlag가 0인 경우
        ringFlag = -1;
        // ringFlag를 -1로 설정합니다.
        for (int i = 0; i < (int)laserCloudMsg->fields.size(); ++i)
        {
            // laserCloudMsg의 필드 수 만큼 반복하면서
            if (laserCloudMsg->fields[i].name == "ring")
            {
                // 필드의 이름이 "ring"인 경우
                ringFlag = 1;
                // ringFlag를 1로 설정하고 반복문을 종료합니다.
                break;
            }
        }
        // ringFlag가 -1인 경우
        if (ringFlag == -1)
        {
            // "Point cloud ring channel not available, please configure your point cloud data!" 오류 메시지를 출력하고 함수를 종료합니다.
            cout << "Point cloud ring channel not available, please configure your point cloud data!" << endl;
            return;
        }
    }


    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);
    // 입력 포인트 클라우드에서 최소 거리 임계값(MINIMUM_RANGE)보다 가까운 포인트를 제거합니다.
    
    int cloudSize = laserCloudIn.points.size();
    // 포인트 클라우드의 크기를 가져와서 변수 cloudSize에 저장합니다.
    
    // 'intensity' 값을 필터링합니다.
    for(int i = 4; i < (cloudSize - 4); i++) 
    {
        // 포인트 클라우드의 4번째 포인트부터 끝에서 4번째 포인트까지 반복합니다.
        // 현재 포인트의 'intensity' 값을 주변 9개 포인트의 'intensity' 평균값으로 업데이트합니다.
        laserCloudIn.points[i].intensity = (laserCloudIn.points[i - 4].intensity +
                                            laserCloudIn.points[i - 3].intensity +
                                            laserCloudIn.points[i - 2].intensity +
                                            laserCloudIn.points[i - 1].intensity +
                                            laserCloudIn.points[i].intensity +
                                            laserCloudIn.points[i + 1].intensity +
                                            laserCloudIn.points[i + 2].intensity +
                                            laserCloudIn.points[i + 3].intensity +
                                            laserCloudIn.points[i + 4].intensity) / 9;
    }
    // 포인트 클라우드의 'intensity' 값을 필터링하여 업데이트합니다.

      int scanID = 0;
    // 스캔 ID를 나타내는 변수를 초기화합니다.
    
    PointType point;
    // 포인트 클라우드에서 사용할 포인트 객체를 생성합니다.
    
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    // N_SCANS 개수만큼의 포인트 클라우드를 저장할 벡터인 laserCloudScans를 생성합니다.
    
    for (int i = 0; i < cloudSize; i++)
    {
        // 포인트 클라우드의 크기만큼 반복합니다.
    
        // 포인트 객체의 좌표와 강도 값을 입력 포인트 클라우드(laserCloudIn)에서 가져옵니다.
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;
        point.intensity = laserCloudIn.points[i].intensity;
    
        // 현재 포인트의 스캔 ID를 가져옵니다.
        scanID = laserCloudIn.points[i].ring;
    
        // 포인트의 강도에 스캔 ID를 더합니다.
        point.intensity += scanID;
    
        // 현재 포인트를 해당 스캔 ID의 포인트 클라우드(laserCloudScans)에 추가합니다.
        laserCloudScans[scanID].push_back(point);
    }

      
    // int cloudSize = laserCloudIn.points.size();
    // float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    // float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
    //                       laserCloudIn.points[cloudSize - 1].x) +
    //                2 * M_PI;

    // if (endOri - startOri > 3 * M_PI)
    // {
    //     endOri -= 2 * M_PI;
    // }
    // else if (endOri - startOri < M_PI)
    // {
    //     endOri += 2 * M_PI;
    // }
    // //printf("end Ori %f\n", endOri);

    // bool halfPassed = false;
    // int count = cloudSize;
    // PointType point;
    // std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    // for (int i = 0; i < cloudSize; i++)
    // {
    //     point.x = laserCloudIn.points[i].x;
    //     point.y = laserCloudIn.points[i].y;
    //     point.z = laserCloudIn.points[i].z;
    //     point.intensity = laserCloudIn.points[i].intensity / 255.0;

    //     float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
    //     int scanID = 0;

    //     if (N_SCANS == 16)
    //     {
    //         scanID = int((angle + 15) / 2 + 0.5);
    //         if (scanID > (N_SCANS - 1) || scanID < 0)
    //         {
    //             count--;
    //             continue;
    //         }
    //     }
    //     else if (N_SCANS == 32)
    //     {
    //         scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
    //         if (scanID > (N_SCANS - 1) || scanID < 0)
    //         {
    //             count--;
    //             continue;
    //         }
    //     }
    //     else if (N_SCANS == 64)
    //     {   
    //         if (angle >= -8.83)
    //             scanID = int((2 - angle) * 3.0 + 0.5);
    //         else
    //             scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

    //         // use [0 50]  > 50 remove outlies 
    //         if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
    //         {
    //             count--;
    //             continue;
    //         }
    //     }
    //     else
    //     {
    //         printf("wrong scan number\n");
    //         ROS_BREAK();
    //     }
    //     //printf("angle %f scanID %d \n", angle, scanID);

    //     float ori = -atan2(point.y, point.x);
    //     if (!halfPassed)
    //     { 
    //         if (ori < startOri - M_PI / 2)
    //         {
    //             ori += 2 * M_PI;
    //         }
    //         else if (ori > startOri + M_PI * 3 / 2)
    //         {
    //             ori -= 2 * M_PI;
    //         }

    //         if (ori - startOri > M_PI)
    //         {
    //             halfPassed = true;
    //         }
    //     }
    //     else
    //     {
    //         ori += 2 * M_PI;
    //         if (ori < endOri - M_PI * 3 / 2)
    //         {
    //             ori += 2 * M_PI;
    //         }
    //         else if (ori > endOri + M_PI / 2)
    //         {
    //             ori -= 2 * M_PI;
    //         }
    //     }

    //     float relTime = (ori - startOri) / (endOri - startOri);
    //     point.intensity += scanID;
    //     laserCloudScans[scanID].push_back(point); 
    // }

      
    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
  // PointType 유형의 포인트 클라우드를 가리키는 포인터인 laserCloud를 생성합니다.
  
  std::vector<int> laserScanSize(N_SCANS);
  // N_SCANS 개수만큼의 스캔 크기를 저장할 정수 벡터 laserScanSize를 생성합니다.
  
  for (int i = 0; i < N_SCANS; i++)
  { 
      // N_SCANS 개수만큼 반복합니다.
  
      // 현재 스캔의 시작 인덱스를 설정합니다.
      scanStartInd[i] = laserCloud->size() + 5;
  
      // 현재 스캔의 포인트 수를 가져와서 laserScanSize에 저장합니다.
      laserScanSize[i] = laserCloudScans[i].points.size();
  
      // 현재 스캔의 포인트 클라우드를 전체 포인트 클라우드(laserCloud)에 추가합니다.
      *laserCloud += laserCloudScans[i];
  
      // 현재 스캔의 끝 인덱스를 설정합니다.
      scanEndInd[i] = laserCloud->size() - 6;
  }


      
  for (int i = 5; i < cloudSize - 5; i++)
  {
      // 현재 포인트를 중심으로 주변 포인트의 차이를 계산합니다.
      float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
      float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
      float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
      float diffI = laserCloud->points[i - 5].intensity + laserCloud->points[i - 4].intensity + laserCloud->points[i - 3].intensity + laserCloud->points[i - 2].intensity + laserCloud->points[i - 1].intensity - 10 * laserCloud->points[i].intensity + laserCloud->points[i + 1].intensity + laserCloud->points[i + 2].intensity + laserCloud->points[i + 3].intensity + laserCloud->points[i + 4].intensity + laserCloud->points[i + 5].intensity;
  
      // 포인트의 곡률을 계산하고 저장합니다.
      cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
  
      // 포인트의 강도 차이를 계산하고 저장합니다.
      cloudIntensityDiff[i] = diffI * diffI;
  
      // 포인트의 정렬 인덱스를 설정합니다.
      cloudSortInd[i] = i;
  
      // 포인트의 이웃이 선택되었는지 여부를 나타내는 변수를 초기화합니다.
      cloudNeighborPicked[i] = 0;
  
      // 포인트의 레이블을 초기화합니다.
      cloudLabel[i] = 0;
  }


    pcl::PointCloud<PointType> cornerPointsSharp;
    // 날카로운 코너 포인트를 저장할 포인트 클라우드
    
    pcl::PointCloud<PointType> cornerPointsLessSharp;
    // 날카로운 코너 포인트 중 낮은 날카로움을 가진 포인트를 저장할 포인트 클라우드
    
    pcl::PointCloud<PointType> surfPointsFlat;
    // 평평한 표면 포인트를 저장할 포인트 클라우드
    
    pcl::PointCloud<PointType> surfPointsLessFlat;
    // 평평한 표면 포인트 중 낮은 평평함을 가진 포인트를 저장할 포인트 클라우드
    
    pcl::PointCloud<PointType> texturePointSharp;
    // 텍스처 정보가 있는 날카로운 포인트를 저장할 포인트 클라우드
    
    pcl::PointCloud<PointType> texturePointLessSharp;
    // 텍스처 정보가 있는 날카로운 포인트 중 낮은 날카로움을 가진 포인트를 저장할 포인트 클라우드
    
    pcl::PointCloud<PointType> NDTPoints;
    // Normal Distributions Transform (NDT) 포인트를 저장할 포인트 클라우드


  for (int i = 0; i < N_SCANS; i++)
{
    // 스캔마다 반복합니다.

    // 포인트 수가 6 미만인 스캔은 처리하지 않습니다.
    if (scanEndInd[i] - scanStartInd[i] < 6)
        continue;

    // 현재 스캔에 대한 평평한 포인트 클라우드를 생성합니다.
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);

    for (int j = 0; j < 6; j++)
    {
        // 현재 스캔을 6등분하여 처리합니다.
        int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6;
        int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

        // 포인트를 강도 기준으로 정렬합니다.
        std::sort(cloudSortInd + sp, cloudSortInd + ep + 1, comp_intensity);

        int textureNum = 0;

        for (int k = ep; k >= sp; k--)
        {
            int ind = cloudSortInd[k];

            // 아직 선택되지 않은 이웃 포인트 중 낮은 곡률 및 높은 강도 차이를 가진 포인트를 선택합니다.
            if (cloudNeighborPicked[ind] == 0 &&
                cloudCurvature[ind] < 0.1 &&
                cloudIntensityDiff[ind] >= 0.1)
            {
                // 해당 포인트를 선택하고 레이블을 지정합니다.
                cloudNeighborPicked[ind] = 1;
                textureNum++;

                if (textureNum <= 2)
                {
                    cloudLabel[ind] = 1;
                    texturePointSharp.push_back(laserCloud->points[ind]);
                    texturePointLessSharp.push_back(laserCloud->points[ind]);
                }
                else if (textureNum <= 20)
                {
                    cloudLabel[ind] = 2;
                    texturePointLessSharp.push_back(laserCloud->points[ind]);
                }
                else
                {
                    break;
                }

                // 포인트 주변의 인접 포인트를 선택한 것으로 표시합니다.
                for (int l = 1; l <= 5; l++)
                {
                    if (cloudIntensityDiff[ind + l] < 0.01)
                    {
                        break;
                    }

                    cloudNeighborPicked[ind + l] = 1;
                }
                for (int l = -1; l >= -5; l--)
                {
                    if (cloudIntensityDiff[ind + l] < 0.01)
                    {
                        break;
                    }
                    cloudNeighborPicked[ind + l] = 1;
                }
            }
        }


           // 포인트를 곡률 기준으로 정렬합니다.
          std::sort(cloudSortInd + sp, cloudSortInd + ep + 1, comp_corner);
          
          int largestPickedNum = 0;
          
          for (int k = ep; k >= sp; k--)
          {
              int ind = cloudSortInd[k];
          
              // 아직 선택되지 않은 이웃 포인트 중 높은 곡률을 가진 포인트를 선택합니다.
              if (cloudNeighborPicked[ind] == 0 &&
                  cloudCurvature[ind] > 0.1)
              {
                  // 해당 포인트를 선택하고 레이블을 지정합니다.
                  largestPickedNum++;
          
                  if (largestPickedNum <= 2)
                  {
                      cloudLabel[ind] = 3;
                      cornerPointsSharp.push_back(laserCloud->points[ind]);
                      cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                  }
                  else if (largestPickedNum <= 20)
                  {
                      cloudLabel[ind] = 4;
                      cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                  }
                  else
                  {
                      break;
                  }

                  cloudNeighborPicked[ind] = 1; 
                  // 현재 포인트가 선택되었다고 표시합니다.
                  
                  for (int l = 1; l <= 5; l++)
                  {
                      // 현재 포인트에서 다음 포인트까지의 차이를 계산합니다.
                      float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                      float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                      float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                  
                      // 차이가 일정 값보다 크면 이웃 포인트 선택을 중단합니다.
                      if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                      {
                          break;
                      }
                  
                      // 다음 포인트를 선택한 것으로 표시합니다.
                      cloudNeighborPicked[ind + l] = 1;
                  }
                  
                  for (int l = -1; l >= -5; l--)
                  {
                      // 현재 포인트에서 이전 포인트까지의 차이를 계산합니다.
                      float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                      float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                      float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                  
                      // 차이가 일정 값보다 크면 이웃 포인트 선택을 중단합니다.
                      if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                      {
                          break;
                      }
                  
                      // 이전 포인트를 선택한 것으로 표시합니다.
                      cloudNeighborPicked[ind + l] = 1;
                  }

                }
            }

            int smallestPickedNum = 0;
            
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];
            
                // 아직 선택되지 않은 이웃 포인트 중 낮은 곡률을 가진 포인트를 선택합니다.
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < 0.1)
                {
                    // 해당 포인트를 선택하고 레이블을 지정합니다.
                    cloudLabel[ind] = -1;
                    surfPointsFlat.push_back(laserCloud->points[ind]);
            
                    smallestPickedNum++;
            
                    if (smallestPickedNum >= 4)
                    {
                        // 4개 이상의 낮은 곡률 포인트를 선택하면 선택을 중단합니다.
                        break;
                    }
            
                    // 포인트를 선택한 것으로 표시합니다.
                    cloudNeighborPicked[ind] = 1;

                  for (int l = 1; l <= 5; l++)
                  {
                      // 현재 포인트에서 다음 포인트까지의 차이를 계산합니다.
                      float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                      float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                      float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                  
                      // 차이가 일정 값보다 크면 이웃 포인트 선택을 중단합니다.
                      if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                      {
                          break;
                      }
                  
                      // 다음 포인트를 선택한 것으로 표시합니다.
                      cloudNeighborPicked[ind + l] = 1;
                  }
                  
                  for (int l = -1; l >= -5; l--)
                  {
                      // 현재 포인트에서 이전 포인트까지의 차이를 계산합니다.
                      float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                      float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                      float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                  
                      // 차이가 일정 값보다 크면 이웃 포인트 선택을 중단합니다.
                      if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                      {
                          break;
                      }
                  
                      // 이전 포인트를 선택한 것으로 표시합니다.
                      cloudNeighborPicked[ind + l] = 1;
                  }
                }
            }
            
            for (int k = sp; k <= ep; k++)
            {
                // 포인트의 레이블이 0 이하인 경우를 확인하여
                // surfPointsLessFlatScan 포인트 클라우드에 추가합니다.
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
            
            // 포인트 클라우드 다운샘플링
            pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
            pcl::VoxelGrid<PointType> downSizeFilter;
            downSizeFilter.setInputCloud(surfPointsLessFlatScan);
            downSizeFilter.setLeafSize(0.4, 0.4, 0.4);
            downSizeFilter.filter(surfPointsLessFlatScanDS);
            
            // surfPointsLessFlat 포인트 클라우드에 다운샘플링된 포인트 추가
            surfPointsLessFlat += surfPointsLessFlatScanDS;
    }
  
    auto t2 = ros::WallTime::now();
std::cout << "lidar feature extraction: " << (t2 - t1).toSec() * 1000 << "  ms" << std::endl;

// NDTPoints += cornerPointsLessSharp;
// NDTPoints += surfPointsLessFlat;

// Kd 트리를 포인트 클라우드에 설정합니다.
kdtreePlanarCloud->setInputCloud(laserCloud);

// cornerPointsSharp 포인트 클라우드의 각 포인트에 대해 반복합니다.
for(size_t i=0; i<cornerPointsSharp.points.size(); i++) 
{
    // 현재 포인트에서 반경 1.0 내의 포인트 검색
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    kdtreePlanarCloud->radiusSearch(cornerPointsSharp.points[i], 1.0, pointSearchIndLoop, pointSearchSqDisLoop, 20);

    // 검색된 포인트에 대해 레이블이 0 이상인 포인트만 선택하여 NDTPoints에 추가
    for(size_t j=0; j<pointSearchIndLoop.size(); j++) 
    {
        if(cloudLabel[pointSearchIndLoop[j]] <= 0) 
            continue;

        NDTPoints.push_back(laserCloud->points[pointSearchIndLoop[j]]);
        cloudLabel[pointSearchIndLoop[j]] = -2;
    }
}

    // surfPointsLessFlat 포인트 클라우드를 NDTPoints에 추가합니다.
    NDTPoints += surfPointsLessFlat;
    cout << NDTPoints.points.size() << endl;
    
    // cornerPointsSharp 포인트 클라우드를 ROS 메시지로 변환하여 퍼블리시합니다.
    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "/map";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);
    
    // cornerPointsLessSharp 포인트 클라우드를 ROS 메시지로 변환하여 퍼블리시합니다.
    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "/map";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);
    
    // surfPointsFlat 포인트 클라우드를 ROS 메시지로 변환하여 퍼블리시합니다.
    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "/map";
    pubSurfPointsFlat.publish(surfPointsFlat2);
    
    // surfPointsLessFlat 포인트 클라우드를 ROS 메시지로 변환하여 퍼블리시합니다.
    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlat2.header.frame_id = "/map";
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);
    
    // texturePointSharp 포인트 클라우드를 ROS 메시지로 변환하여 퍼블리시합니다.
    sensor_msgs::PointCloud2 texturePointSharp2;
    pcl::toROSMsg(texturePointSharp, texturePointSharp2);
    texturePointSharp2.header.stamp = laserCloudMsg->header.stamp;
    texturePointSharp2.header.frame_id = "/map";
    pubTexturePointSharp.publish(texturePointSharp2);
    
    // texturePointLessSharp 포인트 클라우드를 ROS 메시지로 변환하여 퍼블리시합니다.
    sensor_msgs::PointCloud2 texturePointLessSharp2;
    pcl::toROSMsg(texturePointLessSharp, texturePointLessSharp2);
    texturePointLessSharp2.header.stamp = laserCloudMsg->header.stamp;
    texturePointLessSharp2.header.frame_id = "/map";
    pubTexturePointLessSharp.publish(texturePointLessSharp2);
    
    // NDTPoints 포인트 클라우드를 ROS 메시지로 변환하여 퍼블리시합니다.
    sensor_msgs::PointCloud2 NDTPoints2;
    pcl::toROSMsg(NDTPoints, NDTPoints2);
    NDTPoints2.header.stamp = laserCloudMsg->header.stamp;
    NDTPoints2.header.frame_id = "/map";
    PubNDTPoints.publish(NDTPoints2);

}

int main(int argc, char **argv)
{
    // ROS 노드를 초기화합니다.
    ros::init(argc, argv, "scanRegistration");
    ros::NodeHandle nh;

    // ROS 파라미터로부터 스캔 라인 수와 최소 거리를 읽어옵니다.
    nh.param<int>("scan_line", N_SCANS, 64);
    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);

    printf("scan line number %d \n", N_SCANS);

    // 지원하는 스캔 라인 수를 확인하고, 16, 32, 64 중 하나여야 합니다.
    if (N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan lines!");
        return 0;
    }

    // "/points_raw" 토픽에서 포인트 클라우드 데이터를 구독합니다.
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/points_raw", 5, laserCloudHandler);

    // 다양한 필터링된 포인트 클라우드를 다음 토픽에 퍼블리시합니다.
    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 5);
    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 5);
    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 5);
    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 5);
    pubTexturePointSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_texture_sharp", 5);
    pubTexturePointLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_texture_less_sharp", 5);
    PubNDTPoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_NDT", 5);

    // ROS 스핀을 시작하여 노드를 실행합니다.
    ros::spin();

    return 0;
}
