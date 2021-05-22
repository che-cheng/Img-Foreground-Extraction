cmake_minimum_required( VERSION 3.1 )
project( ExtractorTest )
message("===== Img Foreground Extraction Tester =====") 

add_executable(fe_tester fe_tester.cpp)
target_link_libraries(fe_tester
  PUBLIC
    ImgForegroundExtraction
)