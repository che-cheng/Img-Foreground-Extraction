#include "fore_extractor.hpp"

fore_extractor::fore_extractor() { default_parameters(); }

fore_extractor::fore_extractor(const config& config) { cfg_ = config; }

// fore_extractor::~fore_extractor() {}

void fore_extractor::default_parameters() {
  for (double i = 0; i <= 20; i++)
    cfg_.adapt_thresh_prob_bins.push_back(i * 0.05);

  cv::Mat lookUpTable(1, 256, CV_8U);
  uchar* p = lookUpTable.data;
  for (int i = 0; i < 256; ++i) p[i] = uchar(i / (256 / cfg_.num_bins_));
  cfg_.bin_mapping_ = lookUpTable;
}

cv::Mat fore_extractor::extract(const cv::Mat frame, const cv::Rect rect) {
  image_ = frame.clone();
  width_ = frame.cols;
  height_ = frame.rows;

  rect_ = rect;

  GetForegroundBackgroundProbs();
  GetAdaptiveTh();
  OutputResult();
  return result_;
}

void fore_extractor::GetForegroundBackgroundProbs() {
  int imgCount = 1;
  const int channels[] = {0, 1, 2};
  cv::Mat mask = cv::Mat();
  int dims = 3;
  const int sizes[] = {cfg_.num_bins_, cfg_.num_bins_, cfg_.num_bins_};
  float bRange[] = {0, 256};
  float gRange[] = {0, 256};
  float rRange[] = {0, 256};
  const float* ranges[] = {bRange, gRange, rRange};

  cv::Mat surr_hist, obj_hist;
  cv::calcHist(&image_, imgCount, channels, mask, surr_hist, dims, sizes,
               ranges);

  int obj_col = round(rect_.x);
  int obj_row = round(rect_.y);
  int obj_width = round(rect_.width);
  int obj_height = round(rect_.height);

  // if rect size is out of the image boundary
  if ((obj_col + obj_width) > (image_.cols - 1))
    obj_width = (image_.cols - 1) - obj_col;
  if ((obj_row + obj_height) > (image_.rows - 1))
    obj_height = (image_.rows - 1) - obj_row;

  cv::Mat obj_win;
  cv::Rect obj_region(std::max(0, obj_col), std::max(0, obj_row),
                      obj_col + obj_width + 1 - std::max(0, obj_col),
                      obj_row + obj_height + 1 - std::max(0, obj_row));
  obj_win = image_(obj_region);
  cv::calcHist(&obj_win, imgCount, channels, mask, obj_hist, dims, sizes,
               ranges);

  // Equation (2) in the paper
  prob_lut_ = (obj_hist + 1.) / (surr_hist + 2.);

  prob_map_ = cv::Mat(image_.size(), CV_32FC1);
  cv::Mat frame_bin;
  cv::LUT(image_, cfg_.bin_mapping_, frame_bin);

  float* p_prob_map = prob_map_.ptr<float>(0);
  cv::MatIterator_<cv::Vec3b> it, end;
  for (it = frame_bin.begin<cv::Vec3b>(), end = frame_bin.end<cv::Vec3b>();
       it != end; ++it) {
    *p_prob_map++ = prob_lut_.at<float>((*it)[0], (*it)[1], (*it)[2]);
  }
}

void fore_extractor::GetAdaptiveTh() {
  rect_.width++;
  rect_.width = std::min(prob_map_.cols - rect_.x, rect_.width);
  rect_.height++;
  rect_.height = std::min(prob_map_.rows - rect_.y, rect_.height);
  cv::Mat obj_prob_map = prob_map_(rect_);
  int bins = 21;
  float range[] = {-0.025, 1.025};
  const float* histRange = {range};
  bool uniform = true;
  bool accumulate = false;

  cv::Mat H_obj, H_dist;
  /// Compute the histograms:
  cv::calcHist(&obj_prob_map, 1, 0, cv::Mat(), H_obj, 1, &bins, &histRange,
               uniform, accumulate);

  H_obj = H_obj / cv::sum(H_obj)[0];
  cv::Mat cum_H_obj = H_obj.clone();
  for (int i = 1; i < cum_H_obj.rows; ++i)
    cum_H_obj.at<float>(i, 0) += cum_H_obj.at<float>(i - 1, 0);

  cv::calcHist(&prob_map_, 1, 0, cv::Mat(), H_dist, 1, &bins, &histRange,
               uniform, accumulate);
  H_dist = H_dist - H_obj;
  H_dist = H_dist / cv::sum(H_dist)[0];
  cv::Mat cum_H_dist = H_dist.clone();
  for (int i = 1; i < cum_H_dist.rows; ++i)
    cum_H_dist.at<float>(i, 0) += cum_H_dist.at<float>(i - 1, 0);

  cv::Mat k(cum_H_obj.size(), cum_H_obj.type(), cv::Scalar(0.0));
  for (int i = 0; i < (k.rows - 1); ++i)
    k.at<float>(i, 0) =
        cum_H_obj.at<float>(i + 1, 0) - cum_H_obj.at<float>(i, 0);
  cv::Mat cum_H_obj_lt = (cum_H_obj < (1 - cum_H_dist));
  cum_H_obj_lt.convertTo(cum_H_obj_lt, CV_32FC1, 1.0 / 255);
  cv::Mat x = abs(cum_H_obj - (1 - cum_H_dist)) + cum_H_obj_lt + (1 - k);
  float xmin = 100;
  int min_index = 0;
  for (int i = 0; i < x.rows; ++i) {
    if (xmin > x.at<float>(i, 0)) {
      xmin = x.at<float>(i, 0);
      min_index = i;
    }
  }

  // Final threshold result should lie between 0.4 and 0.7 to be not too
  // restrictive
  threshold_ =
      std::max(.4, std::min(.7, cfg_.adapt_thresh_prob_bins[min_index]));
}

void fore_extractor::OutputResult() {
  result_ = cv::Mat::zeros(cv::Size(width_, height_), CV_8UC3);
  for (int i = rect_.tl().y + 1; i < rect_.br().y; i++) {
    float* prob_i = prob_map_.ptr<float>(i);
    for (int j = rect_.tl().x + 1; j < rect_.br().x; j++) {
      float prob_val = prob_i[j];
      if(prob_val > threshold_ + 0.15) {
        result_.at<cv::Vec3b>(i,j) = image_.at<cv::Vec3b>(i,j);
      }
    }
  }
}