#include"slambase.h"



PointCloud::Ptr imageToPointCloud(cv::Mat &rgb,cv::Mat &depth,CAMERA_INTRINSIC_PARAMETERS &cam)
{
  PointCloud::Ptr cloud (new PointCloud);
  for(int m=0;m<depth.rows;m++)
   {   
    for(int n=0;n<depth.cols;n++)
       {
           ushort d=depth.ptr<ushort>(m)[n];
           if(d==0)
           continue;
          
           PointT p;
         p.z=double(d);
         p.x=(n-cam.cx)*p.z/cam.fx;
         p.y=(m-cam.cy)*p.z/cam.fy;

         p.b=rgb.ptr<uchar>(m)[n*3];
         p.g=rgb.ptr<uchar>(m)[n*3+1];
         p.r=rgb.ptr<uchar>(m)[n*3+2];

         cloud->points.push_back(p);
         }
    }
      
   cloud->height=1;
   cloud->width=cloud->points.size();
   cloud->is_dense=false;
   return cloud;


}
PointCloud::Ptr imageToPointCloud(cv::Mat &rgb,cv::Mat &depth,cv::Mat &cam)
{
  PointCloud::Ptr cloud (new PointCloud);
  for(int m=0;m<depth.rows;m++)
   {
    for(int n=0;n<depth.cols;n++)
       {
           ushort d=depth.ptr<ushort>(m)[n];
           if(d==0)
           continue;

           PointT p;
         p.z=double(d);
         p.x=(n-cam.at<double>(0,3))*p.z/cam.at<double>(0.0);
         p.y=(m-cam.at<double>(1,3))*p.z/cam.at<double>(1,2);

         p.b=rgb.ptr<uchar>(m)[n*3];
         p.g=rgb.ptr<uchar>(m)[n*3+1];
         p.r=rgb.ptr<uchar>(m)[n*3+2];

         cloud->points.push_back(p);
         }
    }

   cloud->height=1;
   cloud->width=cloud->points.size();
   cloud->is_dense=false;
   return cloud;


}

cv::Point3f Point2dTo3d(cv::Point3f &point,CAMERA_INTRINSIC_PARAMETERS &cam)
{
  cv::Point3f p;
  p.z=double(point.z);
  p.x=(point.x-cam.cx)*p.z/cam.fx;
  p.y=(point.y-cam.cy)*p.z/cam.fy;
  return p;

}
cv::Point2d pixel2cam ( const cv::Point2d& p, const cv::Mat& K )
{
    return cv::Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}
