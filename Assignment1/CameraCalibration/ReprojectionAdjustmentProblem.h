#pragma once

#include <vector>

#include <opencv2/core/core.hpp>

cv::Vec3f reprojectionAdjustment(
	const std::vector<cv::Mat1f>& homographies,
	const std::vector<cv::Vec2f>& points,
	const cv::Vec3f& initialGuess
);
