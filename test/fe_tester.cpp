#include <iostream>

#include "fore_extractor.hpp"
#include "util/util.hpp"

int main() {
  
  cv::Mat src = cv::imread("../test/test_img/stop_sign.jpg");
  int width = src.cols;
  int height = src.rows;
  cv::Rect obj_rect_surr = cv::Rect(cv::Point2i(230,190), cv::Point2i(1200,1280));

  /* main function */
  fore_extractor extractor;
  cv::Mat result = extractor.extract(src, obj_rect_surr);
  /* main function end */
  cv::Mat color_prob = SudoColor(extractor.GetProbMap());

  /* show result */
  cv::rectangle(src, obj_rect_surr, cv::Vec3b(0, 255, 0), 5);
  cv::Mat debug_img_ = cv::Mat(cv::Size(width * 3, height), CV_8UC3);
  src.copyTo(debug_img_(cv::Rect(0, 0, width, height)));
  color_prob.copyTo(debug_img_(cv::Rect(width, 0, width, height)));
  result.copyTo(debug_img_(cv::Rect(width * 2, 0, width, height)));
  cv::imshow("Result", debug_img_);
  cv::waitKey(0);
}