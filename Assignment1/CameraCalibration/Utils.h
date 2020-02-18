#pragma once

#include <QImage>
#include <QVector3D>

#include <opencv2/core/core.hpp>

cv::Vec2f projectPoint(const cv::Mat1f& H, const cv::Vec3f& m);
cv::Vec2f projectPoint(const cv::Mat1f& H, const cv::Vec4f& m);

cv::Mat computeProjectionMatrix(cv::Mat cameraMatrix, cv::Mat rvec, cv::Mat tvec);

// The 3x4 projection matrix without the 3rd column and normalized
cv::Mat1f removeZProjectionMatrix(const cv::Mat& projectionMatrix);

cv::Mat rotationX180(const cv::Mat1f& matrix);

bool isRotationMatrix(const cv::Mat1f& R);

cv::Vec3f rotationMatrixToEulerAngles(const cv::Mat1f& R);

cv::Vec3f rotationMatrixToEulerAnglesDeg(const cv::Mat1f& R);

std::vector<cv::Mat> convert(const std::vector<cv::Mat1f>& v);

double computeRMSReProjectionError(
	const std::vector<std::vector<cv::Vec3f>>& objectPoints,
	const std::vector<std::vector<cv::Vec2f>>& imagePoints,
	const cv::Mat& cameraMatrix,
	const cv::Mat& distCoeffs,
	const std::vector<cv::Mat>& rvecs,
	const std::vector<cv::Mat>& tvecs);

double computeRMSReProjectionError(
	const std::vector<std::vector<cv::Vec3f>>& objectPoints,
	const std::vector<std::vector<cv::Vec2f>>& imagePoints,
	const cv::Mat& cameraMatrix,
	const cv::Mat& distCoeffs,
	const std::vector<cv::Mat1f>& rvecs,
	const std::vector<cv::Mat1f>& tvecs);

double computeAvgReProjectionError(
	const std::vector<std::vector<cv::Vec3f>>& objectPoints,
	const std::vector<std::vector<cv::Vec2f>>& imagePoints,
	const cv::Mat& cameraMatrix,
	const cv::Mat& distCoeffs,
	const std::vector<cv::Mat>& rvecs,
	const std::vector<cv::Mat>& tvecs);

double computeAvgReProjectionError(
	const std::vector<std::vector<cv::Vec3f>>& objectPoints,
	const std::vector<std::vector<cv::Vec2f>>& imagePoints,
	const cv::Mat& cameraMatrix,
	const cv::Mat& distCoeffs,
	const std::vector<cv::Mat1f>& rvecs,
	const std::vector<cv::Mat1f>& tvecs);

double computeAvgReProjectionError(
	const std::vector<cv::Vec3f>& objectPoints,
	const std::vector<cv::Vec2f>& imagePoints,
	const cv::Mat& H);

std::pair<float, float> focalLengthInMm(const cv::Mat1f& cameraMatrix, const cv::Size& imageSize, const cv::Size2f& sensorSize);

QImage convertToQtImage(cv::InputArray input);

QVector3D convertToQt(const cv::Vec3f& v);

cv::Mat translateImage(const cv::Mat& image, float x, float y);

float getImageCenterX(const cv::Mat1f& cameraMatrix);

float getImageCenterY(const cv::Mat1f& cameraMatrix);
