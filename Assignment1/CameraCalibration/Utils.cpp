#include "Utils.h"

#include <opencv2/calib3d.hpp>

cv::Vec2f projectPoint(const cv::Mat1f& H, const cv::Vec3f& m)
{
	cv::Mat1f point(3, 1);
	point.at<float>(0) = m(0);
	point.at<float>(1) = m(1);
	point.at<float>(2) = 1.0f;

	cv::Mat1f result = H * point;

	// Divide by W
	return cv::Vec2f(
		result.at<float>(0) / result.at<float>(2),
		result.at<float>(1) / result.at<float>(2)
	);
}

// Source: https://answers.opencv.org/question/162932/create-a-stereo-projection-matrix-using-rvec-and-tvec/
cv::Mat computeProjectionMatrix(cv::Mat cameraMatrix, cv::Mat rvec, cv::Mat tvec)
{
	cv::Mat rotMat(3, 3, CV_64F), rotTransMat(3, 4, CV_64F);
	// Convert rotation vector into rotation matrix 
	cv::Rodrigues(rvec, rotMat);
	// Append translation vector to rotation matrix
	cv::hconcat(rotMat, tvec, rotTransMat);
	// Compute projection matrix by multiplying intrinsic parameter 
	// matrix (A) with 3 x 4 rotation and translation pose matrix (RT).
	// Formula: Projection Matrix = A * RT;
	return (cameraMatrix * rotTransMat);
}

cv::Mat1f removeZProjectionMatrix(cv::Mat projectionMatrix)
{
	cv::Mat1f H(3, 3);
	H.col(0) = projectionMatrix.col(0);
	H.col(1) = projectionMatrix.col(1);
	H.col(2) = projectionMatrix.col(3);
	H /= H.at<float>(2, 2);

	return H;
}
