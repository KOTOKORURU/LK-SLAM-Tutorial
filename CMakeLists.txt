CMAKE_MINIMUM_REQUIRED(VERSION 2.8)# she ding banben

PROJECT(slam)#gong cheng ming zi

SET(CMAKE_CXX_COMPILER "g++")#she ding bian yiqi
add_compile_options(-std=c++11)
SET( CMAKE_BUILD_TYPE Debug  )

#ke zhi xing wen jian de mu lu

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#she ding ku wen jian mu lu
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#she ding tou wen jian mu lu
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)
#she ding wei lian jie mu lu
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)




#zeng jia zi wen jian jia(yejiushi ,yuan ma de mulu wenjian jia li zai goujian)
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/src)
