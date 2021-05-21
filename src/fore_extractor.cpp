#include "fore_extractor.hpp"

fore_extractor::fore_extractor() { default_parameters(); }

fore_extractor::fore_extractor(const config& config) { cfg = config; }

fore_extractor::~fore_extractor() {}

void fore_extractor::default_parameters() {
  for (double i = 0; i <= 20; i++)
    cfg.adapt_thresh_prob_bins.push_back(i * 0.05);

  cv::Mat lookUpTable(1, 256, CV_8U);
  uchar* p = lookUpTable.data;
  for (int i = 0; i < 256; ++i) p[i] = uchar(i / (256 / cfg.num_bins_));
  cfg.bin_mapping_ = lookUpTable;
}