#include "Utils.h"

#include <iostream>

#include <QtMath>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>

cv::Vec2f projectPoint(const cv::Mat1f& H, const cv::Vec3f& m)
{
	assert(H.rows == 3);
	assert(H.cols == 3);
	
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

cv::Vec2f projectPoint(const cv::Mat1f& H, const cv::Vec4f& m)
{
	assert(H.rows == 3);
	assert(H.cols == 4);

	cv::Mat1f point(4, 1);
	point.at<float>(0) = m[0];
	point.at<float>(1) = m[1];
	point.at<float>(2) = m[2];
	point.at<float>(3) = 1.0f;

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

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
cv::Vec3f rotationMatrixToEulerAngles(const cv::Mat1f& R)
{
	assert(isRotationMatrix(R));

	float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

	bool singular = sy < 1e-6; // If

	float x, y, z;
	if (!singular)
	{
		x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
		y = atan2(-R.at<double>(2, 0), sy);
		z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
	}
	else
	{
		x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
		y = atan2(-R.at<double>(2, 0), sy);
		z = 0;
	}
	
	return { x, y, z };
}

cv::Vec3f rotationMatrixToEulerAnglesDeg(const cv::Mat1f& R)
{
	const auto anglesRad = rotationMatrixToEulerAngles(R);

	return {
		qRadiansToDegrees(anglesRad[0]),
		qRadiansToDegrees(anglesRad[1]),
		qRadiansToDegrees(anglesRad[2])
	};
}

std::vector<cv::Mat> convert(const std::vector<cv::Mat1f>& v)
{
	std::vector<cv::Mat> u;

	u.reserve(v.size());
	for (const auto& m : v)
	{
		u.push_back(m);
	}

	return u;
}

double computeRMSReProjectionError(
	const std::vector<std::vector<cv::Vec3f>>& objectPoints,
	const std::vector<std::vector<cv::Vec2f>>& imagePoints,
	const cv::Mat& cameraMatrix,
	const cv::Mat& distCoeffs,
	const std::vector<cv::Mat>& rvecs,
	const std::vector<cv::Mat>& tvecs)
{
	double meanError = 0.0f;

	for (unsigned int i = 0; i < objectPoints.size(); i++)
	{
		std::vector<cv::Vec2f> projectedPoints;
		cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, projectedPoints);

		meanError += cv::norm(imagePoints[i], projectedPoints, cv::NORM_L2SQR) / projectedPoints.size();
	}

	return std::sqrt(meanError / objectPoints.size());
}

double computeRMSReProjectionError(
	const std::vector<std::vector<cv::Vec3f>>& objectPoints,
	const std::vector<std::vector<cv::Vec2f>>& imagePoints,
	const cv::Mat& cameraMatrix,
	const cv::Mat& distCoeffs,
	const std::vector<cv::Mat1f>& rvecs,
	const std::vector<cv::Mat1f>& tvecs)
{
	return computeRMSReProjectionError(
		objectPoints,
		imagePoints,
		cameraMatrix,
		distCoeffs,
		convert(rvecs),
		convert(tvecs)
	);
}

double computeAvgReProjectionError(
	const std::vector<std::vector<cv::Vec3f>>& objectPoints,
	const std::vector<std::vector<cv::Vec2f>>& imagePoints,
	const cv::Mat& cameraMatrix,
	const cv::Mat& distCoeffs,
	const std::vector<cv::Mat>& rvecs,
	const std::vector<cv::Mat>& tvecs)
{
	double meanError = 0.0f;

	for (unsigned int i = 0; i < objectPoints.size(); i++)
	{
		std::vector<cv::Vec2f> projectedPoints;
		cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, projectedPoints);

		double viewMeanError = 0.0;

		for (unsigned int j = 0; j < projectedPoints.size(); j++)
		{
			viewMeanError += cv::norm(imagePoints[i][j], projectedPoints[j], cv::NORM_L2);
		}

		meanError += viewMeanError / projectedPoints.size();
	}

	return meanError / objectPoints.size();
}

double computeAvgReProjectionError(
	const std::vector<std::vector<cv::Vec3f>>& objectPoints,
	const std::vector<std::vector<cv::Vec2f>>& imagePoints,
	const cv::Mat& cameraMatrix,
	const cv::Mat& distCoeffs,
	const std::vector<cv::Mat1f>& rvecs,
	const std::vector<cv::Mat1f>& tvecs)
{
	return computeAvgReProjectionError(
		objectPoints,
		imagePoints,
		cameraMatrix,
		distCoeffs,
		convert(rvecs),
		convert(tvecs)
	);
}

double computeAvgReProjectionError(
	const std::vector<cv::Vec3f>& objectPoints,
	const std::vector<cv::Vec2f>& imagePoints,
	const cv::Mat& H)
{
	float meanError = 0.0f;
	for (unsigned int i = 0; i < objectPoints.size(); i++)
	{
		const auto projectedPoint = projectPoint(H, objectPoints[i]);
		meanError += cv::norm(imagePoints[i], projectedPoint, cv::NORM_L2);
	}

	return meanError / objectPoints.size();
}

std::pair<float, float> focalLengthInMm(const cv::Mat1f& cameraMatrix, const cv::Size& imageSize, const cv::Size2f& sensorSize)
{
	return {
		cameraMatrix.at<float>(0, 0) * sensorSize.width / imageSize.width,
		cameraMatrix.at<float>(1, 1) * sensorSize.height / imageSize.height
	};
}

QImage convertToQtImage(cv::InputArray input)
{
	assert(input.type() == CV_8UC3);

	const cv::Mat view(input.getMat());
	cv::Mat rgbImage;
	cv::cvtColor(view, rgbImage, cv::COLOR_BGR2RGB);

	const QImage imageView(rgbImage.data, rgbImage.cols, rgbImage.rows, rgbImage.step, QImage::Format_RGB888);

	return imageView.convertToFormat(QImage::Format_RGB32);
}

QVector3D convertToQt(const cv::Vec3f& v)
{
	return {
		v[0],
		v[1],
		v[2]
	};
}

cv::Mat translateImage(const cv::Mat& image, float x, float y)
{
	const cv::Scalar black(0.0, 0.0, 0.0, 255.0);
	const cv::Mat translationMat = (cv::Mat_<double>(2, 3) << 1, 0, x, 0, 1, y);

	cv::Mat translatedImage;
	cv::warpAffine(image,
		           translatedImage,
		           translationMat,
		           image.size(),
		           cv::INTER_LINEAR,
		           cv::BORDER_CONSTANT,
		           black);

	return translatedImage.clone();
}

float getImageCenterX(const cv::Mat1f& cameraMatrix)
{
	return cameraMatrix.at<float>(0, 2);
}

float getImageCenterY(const cv::Mat1f& cameraMatrix)
{
	return cameraMatrix.at<float>(1, 2);
}
