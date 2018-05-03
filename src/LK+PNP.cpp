#include<iostream>
#include<string>
#include"slambase.h"
#include<vector>
#include<chrono>
#include<sophus/so3.h>
#include<sophus/se3.h>
#include"g2o_types.h"

using namespace std;
void bundleAdjustment(const vector<cv::Point3f> pts_3d,const vector<cv::Point2f>pts_2d,const cv::Mat& K,const cv::Mat& R,const cv::Mat& t);
int main(int argc,char** argv)
{
  cv::Mat K=(cv::Mat_<double>(3,3)<<520.9,0,325.1,0,521.0,249.7,0,0,1);

 cv::Mat rgb1,depth1;
 cv::Mat rgb2,depth2;
 rgb1=cv::imread("./data2/1.png",0);
 rgb2=cv::imread("./data2/2.png",0);
 depth1=cv::imread("./data2/1_depth.png",-1);
 //depth2=cv::imread("./data2/1305031102.363013.png",-1);


 vector<cv::Point2f> VecCorners;
 double ql=0.01;
 double minDistance=10.0;
 
 int maxCorners=50;

chrono::steady_clock::time_point t1=chrono::steady_clock::now();
 cv::goodFeaturesToTrack(rgb1,VecCorners,maxCorners,ql,minDistance,cv::Mat(),3,false,0.04); 
 cout<<"corners:"<<VecCorners.size()<<endl;
 //cv::Mat color=rgb1.clone();
 
 //for(auto it=VecCorners.begin();it!=VecCorners.end();it++) 
//{

   //cv::circle(color,*it,10,cv::Scalar(0,240,0),1);
//}
//cv::imshow("corners",color);
//cv::imwrite("./data2/corners.png",color);


//LKS
vector<float> error;
vector<unsigned char>status;
vector<cv::Point2f>next;
vector<cv::Point2f>pre;
for(auto kp:VecCorners)
   pre.push_back(kp);
cv::calcOpticalFlowPyrLK(rgb1,rgb2,pre,next,status,error);
int i=0;
for(auto it=VecCorners.begin();it!=VecCorners.end();i++)
{
    if(status[i]==0)
    {
     it=VecCorners.erase(it);
     continue;
     }
    it++;
}

cout<<VecCorners.size()<<endl;
//cv::Mat img2=rgb2.clone();
//cv::Mat img1=rgb1.clone();

//for(auto kp:VecCorners)
//{
  //cv::circle(img1,kp,10,cv::Scalar(0,240,0),1);
//}
//cv::imshow("afer",img1);
//cv::imwrite("./data2/corners-afer.png",img1);
//for(auto kp:next)
//{
//cv::circle(img2,kp,10,cv::Scalar(0,240,0),1);
//}
//cv::imshow("next",img2);
//cv::imwrite("./data2/next.png",img2);



//PNP
myslam::Camera::Ptr cam(new myslam::Camera);
vector<cv::Point3f>pts_3d;
vector<cv::Point2f>pts_2d;
for(int i=0;i<VecCorners.size();i++)
{
 ushort d=depth1.ptr<unsigned short>(int(VecCorners[i].y))[int(VecCorners[i].x)];
if(d==0)
  continue;
float dd=d/1000.0;
cv::Point2d p1=pixel2cam(VecCorners[i],K);
pts_3d.push_back(cv::Point3f(p1.x*dd,p1.y*dd,dd)); 
pts_2d.push_back(next[i]);
}
cout<<"pts_2d size:"<<pts_2d.size()<<endl;
cv::Mat r,t,inliers;
cv::solvePnPRansac(pts_3d,pts_2d,K,cv::Mat(),r,t,false,100,1.0,0.99,inliers,cv::SOLVEPNP_ITERATIVE);
//cv::Mat R;
//cv::Rodrigues(r,R);
//cout<<"R="<<endl<<R<<endl;
//cout<<"t="<<endl<<t<<endl;
//cout<<"r="<<endl<<r<<endl;
Sophus::SE3 se1(Sophus::SO3(r.at<double>(0,0),r.at<double>(1,0),r.at<double>(2,0)),Eigen::Vector3d(t.at<double>(0,0),t.at<double>(1,0),t.at<double>(2,0)));
//Eigen::Matrix3d r1;
//cv::cv2eigen(R,r1);
//Sophus::SE3 se2(r1,Eigen::Vector3d(t.at<double>(0,0),t.at<double>(1,0),t.at<double>(2,0)));
//cout<<"se1:"<<endl<<se1<<endl;
//cout<<"se2:"<<endl<<se2<<endl;
typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>>Block;
Block::LinearSolverType* linearSolver=new g2o::LinearSolverDense<Block::PoseMatrixType>();
Block* solver_ptr=new Block(linearSolver);
g2o::OptimizationAlgorithmLevenberg* solver=new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
g2o::SparseOptimizer optimizer;
optimizer.setAlgorithm(solver);
//vertex
g2o::VertexSE3Expmap* pose=new g2o::VertexSE3Expmap();
pose->setId(0);
pose->setEstimate(g2o::SE3Quat(se1.rotation_matrix(),se1.translation()));
optimizer.addVertex(pose);
//edges
for(int i;i<inliers.rows;i++)
{
    int index=inliers.at<int>(i,0);
    myslam::EdgeProjectXYZ2UVPoseOnly* edge=new myslam::EdgeProjectXYZ2UVPoseOnly();
    edge->setId(i);
    edge->setVertex(0,pose);
    edge->camera=cam;
    edge->point=Eigen::Vector3d(pts_3d[index].x,pts_3d[index].y,pts_3d[index].z);
    edge->setMeasurement(Eigen::Vector2d(pts_2d[index].x,pts_2d[index].y));
    edge->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edge);

}
optimizer.initializeOptimization();
optimizer.optimize(10);
se1=Sophus::SE3(pose->estimate().rotation(),pose->estimate().translation());
Sophus::SE3 T = Sophus::SE3();
T=se1*T;
Sophus::SE3 Twc=T.inverse();

cout<<Twc<<endl;
Eigen::Matrix3d R=Twc.rotation_matrix();
Eigen::Vector3d t3=Twc.translation();

cout<<"R:"<<R<<endl;
cout<<"t:"<<t3<<endl;


//BA
//bundleAdjustment(pts_3d,pts_2d,K,R,t);
chrono::steady_clock::time_point t2=chrono::steady_clock::now();
chrono::duration<double> time_used=chrono::duration_cast<chrono::duration<double>>(t2-t1);
cout<<"time used:"<<time_used.count()<<endl;
return 0;  

}


void bundleAdjustment(const vector<cv::Point3f> pts_3d,const vector<cv::Point2f>pts_2d,const cv::Mat& K,const cv::Mat& R,const cv::Mat& t)
{
//初始化
typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>>  Block;//pose维度6，3d点维度为3；
Block::LinearSolverType* linearSolver=new g2o::LinearSolverCSparse<Block::PoseMatrixType>();
Block* solve_ptr=new Block(linearSolver);
g2o::OptimizationAlgorithmLevenberg* solver=new g2o::OptimizationAlgorithmLevenberg(solve_ptr);
g2o::SparseOptimizer optimizer;
optimizer.setAlgorithm(solver);

//设置位姿节点

g2o::VertexSE3Expmap* pose=new g2o::VertexSE3Expmap();
Eigen::Matrix3d R_mat;
//旋转举证
R_mat<<R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),
       R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),
       R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2);
pose->setId(0);//第0个顶点
//设置计算值
pose->setEstimate(g2o::SE3Quat(R_mat,Eigen::Vector3d(t.at<double>(0,0),t.at<double>(1,0),t.at<double>(2,0))));
//添加顶点
optimizer.addVertex(pose);

//设置3D点节点
int index=1;
for(const cv::Point3f p:pts_3d)
{
  g2o::VertexSBAPointXYZ* point=new g2o::VertexSBAPointXYZ();
  point->setId(index++);//1～n个顶点
  point->setEstimate(Eigen::Vector3d(p.x,p.y,p.z));
  point->setMarginalized(true);//边缘化，shur消元
  optimizer.addVertex(point);

}
//相机参数
g2o::CameraParameters* cam=new g2o::CameraParameters(K.at<double>(0,0),Eigen::Vector2d(K.at<double>(0,2),K.at<double>(1,2)),0);
cam->setId(0);
optimizer.addParameter(cam);



//设置投影误差边
index=1;
for(cv::Point2f p:pts_2d)
{
 g2o::EdgeProjectXYZ2UV *edge=new g2o::EdgeProjectXYZ2UV();
 edge->setId(index);
 edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index)));//0指的是_vertices[i]的下标，3d点对应0
 edge->setVertex(1,pose);//_vertices[1]，位姿节点
 
 edge->setMeasurement(Eigen::Vector2d(p.x,p.y));//设置UV值
 edge->setParameterId(0,0); 
 edge->setInformation(Eigen::Matrix2d::Identity());
 
 

 optimizer.addEdge(edge);
 index++;
}

//开始迭代优化
optimizer.setVerbose(true);
optimizer.initializeOptimization();
optimizer.optimize(100);
cout<<endl<<"after optimization:"<<endl;
cout<<"T="<<endl<<Eigen::Isometry3d(pose->estimate()).matrix()<<endl;
//cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;


}











