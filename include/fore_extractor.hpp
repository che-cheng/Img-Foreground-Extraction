/**
 * This file is written according to the part of the paper.
 * "In Defense of Color-based Model-free Tracking,"
 * Author: Horst Possegger, Thomas Mauthner and Horst Bischof
 * For more information to see https://github.com/foolwood/DAT
*/

#ifndef FORE_EXTRACTOR_HPP
#define FORE_EXTRACTOR_HPP

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class fore_extractor {
 public:
  struct config {
    explicit config(){};
    bool show_figure_ = false;
    int img_scale_target_diagonal_ = 75;
    int color_space_ = 1;  // 1rgb 2lab 3hsv 4gray
    int num_bins_ = 16;
    cv::Mat bin_mapping_;                        // getBinMapping(cfg.num_bins);
    std::vector<double> adapt_thresh_prob_bins;  // 0:0.05 : 1;
  };
  fore_extractor();
  fore_extractor(const config& config);

  /* main function */
  cv::Mat extract(const cv::Mat frame, const cv::Rect rect);
  const cv::Mat& GetProbMap() const {return prob_map_;}

 private:
  /* data */
  config cfg_;
  
  cv::Mat image_;
  cv::Mat result_;
  cv::Rect rect_;

  cv::Mat prob_lut_;
  cv::Mat prob_map_;

  double threshold_;
  int width_;
  int height_;

  /* funtions */
  void default_parameters();
  void GetForegroundBackgroundProbs();
  void GetAdaptiveTh();
  void OutputResult();
};

#endif