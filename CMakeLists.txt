cmake_minimum_required(VERSION 3.0) 
# Define project name
PROJECT(ImgForegroundExtraction)
message("===== Img Foreground Extraction =====") 


########## CMake Options ##########
option(BUILD_TEST "Generate extractor tests" Off)

#Set the C++ options
SET(CMAKE_CXX_STANDARD 11)

IF(NOT DEFINED CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release")
ENDIF()

# SET MODULE_OUT_PATH: Be aware of CMAKE_CURRENT_BINARY_DIR
IF( NOT DEFINED MODULE_OUT_PATH )
  SET( MODULE_OUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib/ )
ENDIF()
SET( LIBRARY_OUTPUT_PATH ${MODULE_OUT_PATH} )

SET( TARGET_BUILD_FLAGS
  -Wall
  -pthread
  "$<$<CONFIG:Release>:-O3>"
  "$<$<NOT:$<CONFIG:Release>>:-O0>"
  "$<$<NOT:$<CONFIG:Release>>:-ggdb>"
)


# IF the package has been found, several variables will be SET
if( NOT OpenCV_FOUND)
  find_package(OpenCV 4 REQUIRED)
  if( OpenCV_FOUND )
      message(STATUS "Opencv lib is found= " ${OpenCV_INCLUDE_DIRS} )
      message(STATUS "Opencv version= " ${OpenCV_VERSION})
  endif()
endif()


########## Build Target Module  ###########
add_library(${PROJECT_NAME} SHARED 
  src/fore_extractor.cpp
)
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    ${OpenCV_LIBS}
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    include
    ${OpenCV_INCLUDE_DIRS}
)

message(STATUS "-------- Build Img Foreground Extraction (Done) ---------")
if(BUILD_TEST)
  add_subdirectory(./test)
else ()
  message(STATUS "Generate tests by adding \"-DBUILD_TEST=ON\"")
endif()
