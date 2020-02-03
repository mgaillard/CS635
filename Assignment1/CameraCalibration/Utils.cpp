#include "Utils.h"

#include <opencv2/calib3d.hpp>
#include <iostream>

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
	cv::Mat result = (cameraMatrix * rotTransMat);
	return result.clone();
}

cv::Mat1f removeZProjectionMatrix(const cv::Mat& projectionMatrix)
{
	cv::Mat1f H(3, 3, 0.0f);

	H.at<float>(0, 0) = projectionMatrix.at<float>(0, 0);
	H.at<float>(1, 0) = projectionMatrix.at<float>(1, 0);
	H.at<float>(2, 0) = projectionMatrix.at<float>(2, 0);

	H.at<float>(0, 1) = projectionMatrix.at<float>(0, 1);
	H.at<float>(1, 1) = projectionMatrix.at<float>(1, 1);
	H.at<float>(2, 1) = projectionMatrix.at<float>(2, 1);

	H.at<float>(0, 2) = projectionMatrix.at<float>(0, 3);
	H.at<float>(1, 2) = projectionMatrix.at<float>(1, 3);
	H.at<float>(2, 2) = projectionMatrix.at<float>(2, 3);
	
	H /= H.at<float>(2, 2);

	return H;
}

cv::Mat rotationX180(const cv::Mat1f& matrix)
{
	// 180 degrees rotation matrix
	cv::Mat1f R(3, 3, 0.0f);

	R.at<float>(0, 0) = 1.0f;
	R.at<float>(1, 1) = -1.0f;
	R.at<float>(2, 2) = -1.0f;

	const cv::Mat1f rotationResult = R * matrix;

	return rotationResult.clone();
}

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(const cv::Mat1f& R)
{
	cv::Mat Rt;
	transpose(R, Rt);
	cv::Mat shouldBeIdentity = Rt * R;	
	cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());

	return  cv::norm(I, shouldBeIdentity) < 1e-6;
}
