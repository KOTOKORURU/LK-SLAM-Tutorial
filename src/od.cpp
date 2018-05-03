#include<iostream>
#include<string>
#include"slambase.h"
#include<vector>
#include<chrono>
#include<list>
#include<fstream>
#include<Eigen/Core>
#include<Eigen/Geometry>
#include <sophus/se3.h>
#include <sophus/so3.h>
#include"g2o_types.h"
#include<opencv2/viz.hpp>
using namespace std;
int main(int agrc,char** argv)
{
  cv::Mat K=(cv::Mat_<double>(3,3)<<517.3,0,325.1,0,516.5,249.7,0,0,1);


if(agrc!=2)
{
 cout<<"no file path"<<endl;
 return 0;
}
string path_to_dataset=argv[1];
string associate_file=path_to_dataset+"/associate.txt";
ifstream in(associate_file);
ofstream  out("./motion.txt");
int index_of_motion1=0;
int index_of_motion2=1;

vector<string> rgb_files,depth_files;
vector<string> rgb_times,depth_times;
while(!in.eof())
 {
   string rgb_file,rgb_time,depth_file,depth_time;
  
  in>>rgb_time>>rgb_file>>depth_time>>depth_file;
  
  rgb_times.push_back (rgb_time.c_str());
  depth_times.push_back (depth_time.c_str() );
  
  rgb_files.push_back ( path_to_dataset+"/"+rgb_file );
  depth_files.push_back ( path_to_dataset+"/"+depth_file );
if (in.good() == false )
            break;
}
 cv::Mat color,last_color;
 cv::Mat depth,last_depth;
 
 
 vector<cv::Point2f> VecCorners;
 double ql=0.01;
 double minDistance=10.0;
 int maxCorners=50;
 //origin pose
 Sophus::SE3 Tc=Sophus::SE3();
myslam::Camera::Ptr cam(new myslam::Camera);
chrono::steady_clock::time_point t1=chrono::steady_clock::now();

auto rgb_it=rgb_times.begin();
cout<<rgb_files.size();

//visualization
cv::viz::Viz3d vis("VO");
cv::viz::WCoordinateSystem world_coor(1.0),camera_coor(0.5);
cv::Point3d cam_pos(0,-1.0,-1.0),cam_focal_point(0,0,0),cam_y_dir(0,1,0);
cv::Affine3d cam_pose=cv::viz::makeCameraPose(cam_pos,cam_focal_point,cam_y_dir);
vis.setViewerPose(cam_pose);

world_coor.setRenderingProperty(cv::viz::LINE_WIDTH,2.0);
camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH,1.0);
vis.showWidget("World",world_coor);
vis.showWidget("Camera",camera_coor);


for(int index=0;index<rgb_files.size();index++)
{

 
 color=cv::imread(rgb_files[index],0);
 depth=cv::imread(depth_files[index],-1);
if ( color.data == nullptr || depth.data == nullptr )
            break;


if(index == 0)
{//keypoints
  //vector<cv::Point2f> kps;
 cv::goodFeaturesToTrack(color,VecCorners,maxCorners,ql,minDistance,cv::Mat(),3,false,0.04); 
 //cout<<"corners:"<<VecCorners.size()<<endl;
 //out<<"corners:"<<VecCorners.size()<<endl; 
//for(auto &a:kps)
    //VecCorners.push_back(a);
cout<<"corners:"<<VecCorners.size()<<endl;
//out<<"corners:"<<VecCorners.size()<<endl; 
Sophus::SE3 T_o=Tc.inverse();
cv::Affine3d M(cv::Affine3d::Mat3(
                   T_o.rotation_matrix()(0,0),T_o.rotation_matrix()(0,1),T_o.rotation_matrix()(0,2),
                   T_o.rotation_matrix()(1,0),T_o.rotation_matrix()(1,1),T_o.rotation_matrix()(1,2),
                   T_o.rotation_matrix()(2,0),T_o.rotation_matrix()(2,1),T_o.rotation_matrix()(2,2)
                   ),
               cv::Affine3d::Vec3(T_o.rotation_matrix()(0,0),T_o.rotation_matrix()(1,0),T_o.rotation_matrix()(2,0)
                   )
            );
cv::imshow("image",color);
cv::waitKey(1);
vis.setWidgetPose("Camera",M);
vis.spinOnce(1,false);

 last_color=color;
 last_depth=depth;
 continue;
}


//LKS
vector<float> error;
vector<unsigned char>status;
vector<cv::Point2f>next;
//vector<cv::Point2f>pre;
//for(auto &a:VecCorners)
     //pre.push_back(a);

cv::calcOpticalFlowPyrLK(last_color,color,VecCorners,next,status,error);
//cout<<"vec size"<<pre.size()<<endl;
//out<<"vec.size"<<pre.size()<<endl;
cout<<"flow result:"<<next.size()<<endl;
//out<<"flow result:"<<next.size()<<endl;

int i2=0;
for(auto it=VecCorners.begin();it!=VecCorners.end();i2++)
{   
 
    if(status[i2]==0)
    {
      cv::goodFeaturesToTrack(last_color,VecCorners,maxCorners,ql,minDistance,cv::Mat(),3,false,0.04); 
     cv::calcOpticalFlowPyrLK(last_color,color,VecCorners,next,status,error);
     break;
     }
    it++;
    
}
//cout<<"match result:"<<VecCorners.size()<<endl;
//out<<"match result:"<<VecCorners.size()<<endl;

//PNP

vector<cv::Point3f>pts_3d;
vector<cv::Point2f>pts_2d;
//vector<cv::Point2f>temp;
//for(auto &a:VecCorners)
        //temp.push_back(a);
for(auto i=0;i<VecCorners.size();i++)
{
 ushort d=last_depth.ptr<unsigned short>(int(VecCorners[i].y))[int(VecCorners[i].x)];
if(d==0)
  continue;

float dd=d/5000.0;
cv::Point2d p1=pixel2cam(VecCorners[i],K);
pts_3d.push_back(cv::Point3f(p1.x*dd,p1.y*dd,dd)); 
pts_2d.push_back(next[i]);

}

cv::Mat r,t,inliers;
cv::solvePnPRansac(pts_3d,pts_2d,K,cv::Mat(),r,t,false,100,1.0,0.99,inliers,cv::SOLVEPNP_ITERATIVE);
Sophus::SE3 T=Sophus::SE3(Sophus::SO3(r.at<double>(0,0),r.at<double>(1,0),r.at<double>(2,0)),Eigen::Vector3d(t.at<double>(0,0),t.at<double>(1,0),t.at<double>(2,0)));
//Sophus::SE3 Twc=T.inverse();
//cout<<"inliers:"<<inliers<<endl;
//ba
typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>>Block;
Block::LinearSolverType* linearSolver=new g2o::LinearSolverDense<Block::PoseMatrixType>();
Block* solver_ptr=new Block(linearSolver);
g2o::OptimizationAlgorithmLevenberg* solver=new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
g2o::SparseOptimizer optimizer;
optimizer.setAlgorithm(solver);
//vertex
g2o::VertexSE3Expmap* pose=new g2o::VertexSE3Expmap();
pose->setId(0);
pose->setEstimate(g2o::SE3Quat(T.rotation_matrix(),T.translation()));
optimizer.addVertex(pose);
//edges
for(int i=0;i<inliers.rows;i++)
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
T = Sophus::SE3(pose->estimate().rotation(),pose->estimate().translation());
Tc = T*Tc;
Sophus::SE3 Twc=Tc.inverse();
//MOTION
Sophus::SE3 T_motion=T.inverse();
Eigen::Vector3d t_motion=T_motion.translation();
Eigen::Quaterniond q_motion=T_motion.unit_quaternion();
Eigen::Vector4d x_motion=q_motion.coeffs();


//LOCATION
Eigen::Vector3d tr=Twc.translation();
Eigen::Quaterniond q=Twc.unit_quaternion();
Eigen::Vector4d x=q.coeffs();
cout<<"q="<<q.coeffs()<<endl;
cout<<"t="<<t<<endl;

 
if(rgb_it!=rgb_times.end())
{
    cout<<index<<endl;
    if(rgb_it==rgb_times.begin())
    {
        out<<*rgb_it<<endl;
        rgb_it+=1;
   }
 out<<*rgb_it++<<" "<<tr.transpose()<<" "<<x.transpose()<<endl;
 cout<<index+1<<endl;
}
int i1=0;
for(auto it=VecCorners.begin();it!=VecCorners.end();it++)
{

  *it=next[i1];
  i1++;
}


cv::Affine3d M(
    cv::Affine3d::Mat3(
        Twc.rotation_matrix()(0,0),Twc.rotation_matrix()(0,1),Twc.rotation_matrix()(0,2),
        Twc.rotation_matrix()(1,0),Twc.rotation_matrix()(1,1),Twc.rotation_matrix()(1,2),
        Twc.rotation_matrix()(2,0),Twc.rotation_matrix()(2,1),Twc.rotation_matrix()(2,2)
       ),
        cv::Affine3d::Vec3(
            Twc.translation()(0,0),Twc.translation()(1,0),Twc.translation()(2,0)
        )

     );

cv::imshow("image",color);
cv::waitKey(1);
vis.setWidgetPose("Camera",M);
vis.spinOnce(1,false);


last_depth=depth;
last_color=color;


}
		

chrono::steady_clock::time_point t2=chrono::steady_clock::now();
cout<<"yes"<<endl;
chrono::duration<double>time_used=chrono::duration_cast<chrono::duration<double>>(t2-t1);
cout<<"LK used time:"<<time_used.count()<<".second"<<endl;

return 0;  

}
