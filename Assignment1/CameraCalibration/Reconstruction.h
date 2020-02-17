#pragma once

#include <vector>

#include <QVector3D>

#include <opencv2/core/core.hpp>

void findCorners(
	const cv::Mat& image,
	std::vector<std::vector<cv::Vec3f>>& objectPoints,
	std::vector<std::vector<cv::Vec2f>>& imagePoints,
	const std::string& outputKeypoints,
	const cv::Size& patternSize, // interior number of corners
	float squareSize // Space between two corners
);

void drawProjectedCorners(
	const cv::Mat& image,
	const std::vector<cv::Vec3f>& objectPoints,
	const cv::Mat& cameraMatrix,
	const cv::Mat& distCoeffs,
	const cv::Mat1f rvec,
	const cv::Mat1f tvec,
	const std::string& filename
);

cv::Mat1f cameraPose(const cv::Mat1f& rvec, const cv::Mat1f& tvec);

cv::Vec3f cameraPoseVectorX(const cv::Mat1f& rvec);

std::tuple<QVector3D, QVector3D, QVector3D> cameraEyeAtUpFromPose(
	const cv::Mat1f& cameraMatrix,
	const cv::Mat1f& rvec,
	const cv::Mat1f& tvec
);

cv::Vec3f reconstructPointFromViews(
	const std::vector<cv::Mat1f>& homographies,
	const std::vector<cv::Vec2f>& points
);

cv::Vec3f lookAtPoint(const cv::Mat1f& homography, const cv::Mat1f& cameraMatrix);
