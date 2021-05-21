cmake_minimum_required(VERSION 3.0) 
# Define project name
PROJECT(test)
message("===== test environment =====") 

#Set the C++ options
SET(CMAKE_CXX_STANDARD 14)
SET(CXX_EXTENSIONS ON)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)


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

#link_libraries(asan)
add_compile_options(
    # PRIVATE
  # -pg -g3 -fsanitize=leak -O0
  -fsanitize=address -g
)
add_link_options(
  -fsanitize=address
)

find_package(Eigen3 REQUIRED )

# IF the package has been found, several variables will be SET
if( NOT OpenCV_FOUND)
  find_package(OpenCV 4 REQUIRED)
  if( OpenCV_FOUND )
      message( "Opencv lib is found:" ${OpenCV_INCLUDE_DIRS} )
  endif()
endif()

IF(Eigen3_FOUND)
  MESSAGE(STATUS "  Eigen3(v${Eigen3_VERSION}): ${Eigen3_INCLUDE_DIRS}")
ENDIF()


# add_subdirectory(lib/nlohmann-json)

add_executable(FX_tester FX_tester.cpp)
target_link_libraries( FX_tester ${Eigen3_INCLUDE_DIRS} ${OpenCV_LIBS} )
