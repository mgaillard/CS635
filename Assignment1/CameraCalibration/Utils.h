#pragma once

#include <opencv2/core/core.hpp>

cv::Vec2f projectPoint(const cv::Mat1f& H, const cv::Vec3f& m);

cv::Mat computeProjectionMatrix(cv::Mat cameraMatrix, cv::Mat rvec, cv::Mat tvec);

// The 3x4 projection matrix without the 3rd column and normalized
cv::Mat1f removeZProjectionMatrix(const cv::Mat& projectionMatrix);
