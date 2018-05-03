#include"g2o_types.h"

namespace myslam
{
 void EdgeProjectXYZRGBD::computeError()
{
  const g2o::VertexSBAPointXYZ* point=static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
  const g2o::VertexSE3Expmap* pose=static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
  _error=_measurement-pose->estimate().map(point->estimate());//measurement(这张图的2d点)-R×空间点（上一张图的）+T

}

void EdgeProjectXYZRGBD::linearizeOplus()
{
 g2o::VertexSE3Expmap* pose=static_cast<g2o::VertexSE3Expmap*>(_vertices[1]);
 g2o::SE3Quat T(pose->estimate());
 g2o::VertexSBAPointXYZ* point=static_cast<g2o::VertexSBAPointXYZ*>(_vertices[0]);
//空间中的3d点 
Eigen::Vector3d xyz=point->estimate();
 
//映射到相机坐标系下
Eigen::Vector3d xyz_trans=T.map(xyz);
 
double x=xyz_trans[0];
 double y=xyz_trans[1];
 double z=xyz_trans[2];
 //error to space point
 _jacobianOplusXi=-T.rotation().toRotationMatrix();
  
 //已经转换成了相机坐标，因此单个位姿求导就行了
_jacobianOplusXj(0,0)=0;
_jacobianOplusXj(0,1)=-z;
_jacobianOplusXj(0,2)=y;
_jacobianOplusXj(0,3)=-1;
_jacobianOplusXj(0,4)=0;
_jacobianOplusXj(0,5)=0;

_jacobianOplusXj(1,0)=z;
_jacobianOplusXj(1,1)=0;
_jacobianOplusXj(1,2)=-x;
_jacobianOplusXj(1,3)=0;
_jacobianOplusXj(1,4)=-1;
_jacobianOplusXj(1,5)=0;

_jacobianOplusXj(2,0)=-y;
_jacobianOplusXj(2,1)=x;
_jacobianOplusXj(2,2)=0;
_jacobianOplusXj(2,3)=0;
_jacobianOplusXj(2,4)=0;
_jacobianOplusXj(2,5)=-1;

}

void EdgeProjectXYZRGBDPoseOnly::computeError()
{
 const g2o::VertexSE3Expmap* pose=static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
 _error=_measurement-pose->estimate().map(point);

}

void EdgeProjectXYZRGBDPoseOnly::linearizeOplus()
{ 
 const g2o::VertexSE3Expmap* pose=static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
 g2o::SE3Quat T(pose->estimate());
 Eigen::Vector3d xyz_trans=T.map(point);
 double x=xyz_trans[0];
 double y=xyz_trans[1];
 double z=xyz_trans[2];
  

 //单个位姿求导
_jacobianOplusXi(0,0)=0;
_jacobianOplusXi(0,1)=-z;
_jacobianOplusXi(0,2)=y;
_jacobianOplusXi(0,3)=-1;
_jacobianOplusXi(0,4)=0;
_jacobianOplusXi(0,5)=0;

_jacobianOplusXi(1,0)=z;
_jacobianOplusXi(1,1)=0;
_jacobianOplusXi(1,2)=-x;
_jacobianOplusXi(1,3)=0;
_jacobianOplusXi(1,4)=-1;
_jacobianOplusXi(1,5)=0;

_jacobianOplusXi(2,0)=-y;
_jacobianOplusXi(2,1)=x;
_jacobianOplusXi(2,2)=0;
_jacobianOplusXi(2,3)=0;
_jacobianOplusXi(2,4)=0;
_jacobianOplusXi(2,5)=-1;
}

void EdgeProjectXYZ2UVPoseOnly::computeError()
{
  const g2o::VertexSE3Expmap* pose=static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
  _error=_measurement-camera->camera2pixel(pose->estimate().map(point));
}
void EdgeProjectXYZ2UVPoseOnly::linearizeOplus()
{
  const g2o::VertexSE3Expmap* pose=static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
  g2o::SE3Quat T(pose->estimate());
  Eigen::Vector3d xyz_trans=T.map(point);
  double x=xyz_trans[0];
  double y=xyz_trans[1];
  double z=xyz_trans[2];
  double z_2=z*z;
  _jacobianOplusXi(0,0)=x*y/z_2*camera->fx;
  _jacobianOplusXi(0,1)=-camera->fx-(camera->fx*x*x/z_2);
  _jacobianOplusXi(0,2)=camera->fx*y/z;
  _jacobianOplusXi(0,3)=-camera->fx/z;
  _jacobianOplusXi(0,4)=0;
  _jacobianOplusXi(0,5)=camera->fx*x/z_2;

  _jacobianOplusXi(1,0)=camera->fy+(camera->fy*y*y/z_2);
  _jacobianOplusXi(1,1)=-camera->fy*x*y/z_2;
  _jacobianOplusXi(1,2)=-camera->fy*x/z;
  _jacobianOplusXi(1,3)=0;
  _jacobianOplusXi(1,4)=-camera->fy/z;
  _jacobianOplusXi(1,5)=camera->fy*y/z_2;




}
}