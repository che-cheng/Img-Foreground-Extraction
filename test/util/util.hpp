
#ifndef UTIL_HPP
#define UTIL_HPP

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat SudoColor(const cv::Mat &src,
                  const cv::ColormapTypes &type = cv::COLORMAP_JET) {
  cv::Mat norm_src, color_src;
  cv::normalize(src, norm_src, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::applyColorMap(norm_src, color_src, type);
  return color_src;
}

#endif