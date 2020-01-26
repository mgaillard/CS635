#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>

void findCorners(
	const cv::Mat& image,
	std::vector<std::vector<cv::Vec3f>>& objectPoints,
	std::vector<std::vector<cv::Vec2f>>& imagePoints
	)
{
	// Scale the image when looking for corners
	// It speeds up computation and works better
	const float scaleFactor = 0.25;
	const float scaleFactorInverse = 1.0 / scaleFactor;
	// Space between two corners (4 in = 0.1016 m)
	const float squareSize = 0.1016;
	
	// Read image, convert to gray and resize
	cv::Mat imageResized, imageGray, imageGrayResized;
	cv::resize(image, imageResized, cv::Size(0, 0), scaleFactor, scaleFactor);
	cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(imageResized, imageGrayResized, cv::COLOR_BGR2GRAY);

	// Find corners
	const cv::Size patternSize(11, 17); // interior number of corners
	std::vector<cv::Vec2f> corners; // output array for detected corners
	const auto patternFound = cv::findChessboardCorners(imageGrayResized, patternSize, corners);

	// Scale up corners found
	cv::transform(corners, corners, cv::Matx23f(scaleFactorInverse, 0.0, 0.0, 0.0, scaleFactorInverse, 0.0));

	if (patternFound)
	{
		// Refine corners if found
		cornerSubPix(imageGray,
			         corners,
			         cv::Size(11 * scaleFactorInverse, 11 * scaleFactorInverse),
			         cv::Size(-1, -1),
			         cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.1));

		// Display corners in an output file
		// cv::drawChessboardCorners(image, patternSize, cv::Mat(corners), patternFound);
		// cv::imwrite("Images/output.jpg", image);

		// Convert to reference points
		std::vector<cv::Vec3f> currentObjectPoints;

		for (int j = 0; j < patternSize.height; j++)
		{
			for (int i = 0; i < patternSize.width; i++)
			{
				currentObjectPoints.emplace_back(float(i) * squareSize, float(j) * squareSize, 0.0f);
			}
		}

		// Add reference points in the lists
		objectPoints.push_back(currentObjectPoints);
		imagePoints.push_back(corners);
	}
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

float findHomography(
	const std::vector<cv::Vec3f>& objectPoints,
	const std::vector<cv::Vec2f>& imagePoints)
{
	// Project 3D points in 2D
	std::vector<cv::Vec2f> objectPointsPlanar;
	objectPointsPlanar.reserve(objectPoints.size());
	for (unsigned int i = 0; i < objectPoints.size(); i++)
	{
		objectPointsPlanar.emplace_back(objectPoints[i](0), objectPoints[i](1));
	}

	const auto H = cv::findHomography(objectPointsPlanar, imagePoints);
	
	float meanError = 0.0f;
	for (unsigned int i = 0; i < objectPoints.size(); i++)
	{
		const auto projectedPoint = projectPoint(H, objectPoints[i]);
		meanError += cv::norm(imagePoints[i], projectedPoint, cv::NORM_L2);
	}

	return meanError / objectPoints.size();
}

int main(int argc, char* argv[])
{
	const auto imageFront = cv::imread("Images/front.jpg");
	const auto imageLeft = cv::imread("Images/left.jpg");
	
	// Correspondences between 3D points and 2D points in each view 
	std::vector<std::vector<cv::Vec3f>> objectPoints;
	std::vector<std::vector<cv::Vec2f>> imagePoints;
	
	findCorners(imageFront, objectPoints, imagePoints);
	findCorners(imageLeft, objectPoints, imagePoints);

	cv::Mat cameraMatrix;
	cv::Mat distCoeffs;
	std::vector<cv::Mat> rvecs;
	std::vector<cv::Mat> tvecs;
	const auto error = cv::calibrateCamera(objectPoints,
		                                   imagePoints,
		                                   imageFront.size(),
		                                   cameraMatrix,
										   distCoeffs,
		                                   rvecs,
		                                   tvecs);
	
	// Camera matrix M
	std::cout << "Camera matrix = " << std::endl << " " << cameraMatrix << std::endl << std::endl;

	// Distortion
	std::cout << "Distortion coefficients = " << std::endl << " " << distCoeffs << std::endl << std::endl;

	for (unsigned int i = 0; i < objectPoints.size(); i++)
	{
		std::cout << "View " << i << std::endl;

		// Rotation vector
		std::cout << "rvec = " << std::endl << " " << rvecs[i] << std::endl << std::endl;

		// Translation vector
		std::cout << "tvec = " << std::endl << " " << tvecs[i] << std::endl << std::endl;
	}

	// Manually compute the re-projection error
	const auto rmsError = computeRMSReProjectionError(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvecs, tvecs);
	const auto avgError = computeAvgReProjectionError(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvecs, tvecs);
	
	std::cout << "RMS re-projection error = " << rmsError << std::endl;
	std::cout << "AVG re-projection error = " << avgError << std::endl;

	std::cout << findHomography(objectPoints[0], imagePoints[0]) << std::endl;
	std::cout << findHomography(objectPoints[1], imagePoints[1]) << std::endl;

	return 0;
}

/*
#include "MainWindow.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	MainWindow w;
	w.show();
	return a.exec();
}
*/
