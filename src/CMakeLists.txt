FIND_PACKAGE(PCL REQUIRED COMPONENTS common io)
# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04


find_package(Eigen3 REQUIRED)
find_package( OpenCV 3.3 REQUIRED )
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2 
find_package( G2O REQUIRED )
find_package( CSparse REQUIRED )
find_package(Sophus REQUIRED)
find_package(Cholmod REQUIRED)
include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
    ${G2O_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    "/usr/include/eigen3/"
     ${Sophus_INCLUDE_DIR}
    ${CHOLMOD_INCLUDE_DIR}
)



ADD_DEFINITIONS(${PCL_DEFINITIONS})
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
LINK_LIBRARIES(${PCL_LIBRARY_DIRS})


ADD_LIBRARY(SlamBase slambase.cpp)


TARGET_LINK_LIBRARIES(SlamBase
${OpenCV_LIBS}
${PCL_LIBRARIES}

)
ADD_LIBRARY(g2o g2o_types.cpp)
TARGET_LINK_LIBRARIES(g2o
SlamBase
${Sophus_LIBRARIES}

g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
${CSPARSE_LIBRARY}
)
ADD_EXECUTABLE(LK LK+PNP.cpp)
TARGET_LINK_LIBRARIES(LK
SlamBase
g2o
${OpenCV_LIBS}
${PCL_LIBRARIES}
g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
${Sophus_LIBRARIES}
${CSPARSE_LIBRARY}#注意这个不要写错了
)
ADD_EXECUTABLE(od od.cpp)
TARGET_LINK_LIBRARIES(od
SlamBase
g2o
${OpenCV_LIBS}
${PCL_LIBRARIES}
${Sophus_LIBRARY}
)


