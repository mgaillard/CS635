#include "Reconstruction.h"

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "Utils.h"

void findCorners(
	const cv::Mat& image,
	std::vector<std::vector<cv::Vec3f>>& objectPoints,
	std::vector<std::vector<cv::Vec2f>>& imagePoints,
	const std::string& outputKeypoints,
	const cv::Size& patternSize, // interior number of corners
	float squareSize // Space between two corners
)
{
	// Scale the image when looking for corners
	// It speeds up computation and works better
	const float scaleFactor = 0.25;
	const float scaleFactorInverse = 1.0 / scaleFactor;
	

	// Read image, convert to gray and resize
	cv::Mat imageResized, imageGray, imageGrayResized;
	cv::resize(image, imageResized, cv::Size(0, 0), scaleFactor, scaleFactor);
	cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(imageResized, imageGrayResized, cv::COLOR_BGR2GRAY);

	// Find corners
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
		cv::Mat newImage = image.clone();
		cv::drawChessboardCorners(newImage, patternSize, cv::Mat(corners), patternFound);
		cv::imwrite(outputKeypoints, newImage);

		// Convert to reference points
		std::vector<cv::Vec3f> currentObjectPoints;

		for (int j = 0; j < patternSize.height; j++)
		{
			for (int i = 0; i < patternSize.width; i++)
			{
				currentObjectPoints.emplace_back(float(j) * squareSize, float(i) * squareSize, 0.0f);
			}
		}

		// Add reference points in the lists
		objectPoints.push_back(currentObjectPoints);
		imagePoints.push_back(corners);
	}
}

void drawProjectedCorners(
	const cv::Mat& image,
	const std::vector<cv::Vec3f>& objectPoints,
	const cv::Mat& cameraMatrix,
	const cv::Mat& distCoeffs,
	const cv::Mat1f rvec,
	const cv::Mat1f tvec,
	const std::string& filename)
{
	const std::vector<cv::Vec3f> objectPointsX = {
		{0 * 0.1016f, 0.0f, 0.0f},
		{1 * 0.1016f, 0.0f, 0.0f},
		{2 * 0.1016f, 0.0f, 0.0f},
		{3 * 0.1016f, 0.0f, 0.0f},
		{4 * 0.1016f, 0.0f, 0.0f},
		{5 * 0.1016f, 0.0f, 0.0f},
	};


	std::vector<cv::Vec2f> projectedPoints;
	cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

	cv::Mat newImage = image.clone();

	for (const auto p : projectedPoints)
	{
		cv::circle(
			newImage,
			cv::Point(int(p[0]), int(p[1])),
			3,
			cv::Scalar(0, 0, 255),
			cv::FILLED
		);
	}

	cv::imwrite(filename, newImage);
}

void cameraPose(const cv::Mat1f& rvec, const cv::Mat1f& tvec)
{
	cv::Mat1f R;
	cv::Rodrigues(rvec, R); // R is 3x3

	auto invTvec = -R.t() * tvec; // translation of inverse
	R = rotationX180(R.t());  // rotation of inverse

	cv::Mat1f T = cv::Mat1f::eye(4, 4); // T is 4x4
	T(cv::Range(0, 3), cv::Range(0, 3)) = R.t() * 1; // copies R into T
	T(cv::Range(0, 3), cv::Range(3, 4)) = invTvec * 1; // copies tvec into T

	std::cout << "T = " << std::endl << T << std::endl;

	// To get the Euler angles (XYZ)
	// Go to https://www.andre-gaschler.com/rotationconverter/
	// Copy the rotation matrix R
	// Input the angles (in degrees) in blender 
}
