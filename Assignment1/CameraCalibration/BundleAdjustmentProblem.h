#pragma once

#include <opencv2/core/core.hpp>

std::tuple<cv::Mat1f, std::vector<cv::Mat1f>, std::vector<cv::Mat1f>>
bundleAdjustment(
	const std::vector<std::vector<cv::Vec3f>>& objectPoints,
	const std::vector<std::vector<cv::Vec2f>>& imagePoints,
	const cv::Mat1f& cameraMatrix,
	const std::vector<cv::Mat1f>& rvecs,
	const std::vector<cv::Mat1f>& tvecs
);
