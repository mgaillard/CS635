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

cv::Mat1f cameraPose(const cv::Mat1f& rvec, const cv::Mat1f& tvec)
{
	cv::Mat1f R;
	cv::Rodrigues(rvec, R); // R is 3x3

	auto invTvec = -R.t() * tvec; // translation of inverse
	R = rotationX180(R.t());  // rotation of inverse

	cv::Mat1f T = cv::Mat1f::eye(4, 4); // T is 4x4
	T(cv::Range(0, 3), cv::Range(0, 3)) = R.t() * 1; // copies R into T
	T(cv::Range(0, 3), cv::Range(3, 4)) = invTvec * 1; // copies tvec into T

	// To get the Euler angles (XYZ)
	// Go to https://www.andre-gaschler.com/rotationconverter/
	// Copy the rotation matrix R
	// Input the angles (in degrees) in blender
	
	return T;
}

cv::Vec3f cameraPoseVectorX(const cv::Mat1f& rvec)
{
	cv::Mat1f R;
	cv::Rodrigues(rvec, R); // R is 3x3

	// X is the first column of the R transposed matrix
	cv::Mat1f T = R.t() * 1.f; // copies R into T

	return {
		T.at<float>(0, 0),
		T.at<float>(1, 0),
		T.at<float>(2, 0)
	};
}

std::tuple<QVector3D, QVector3D, QVector3D> cameraEyeAtUpFromPose(
	const cv::Mat1f& cameraMatrix,
	const cv::Mat1f& rvec,
	const cv::Mat1f& tvec
)
{
	const auto homography = computeProjectionMatrix(cameraMatrix, rvec, tvec);
	const auto pose = cameraPose(rvec, tvec);

	// Find position of the camera in world coordinates
	const QVector3D eye(
		pose.at<float>(0, 3),
		pose.at<float>(1, 3),
		pose.at<float>(2, 3)
	);

	// Find the look at point (intersection between optical center and plane z=0)
	const auto at = convertToQt(lookAtPoint(homography, cameraMatrix));

	// Get the X vector of the camera in world coordinates
	const QVector3D x = convertToQt(cameraPoseVectorX(rvec));
	// From the X vector, we can get the up vector
	const QVector3D eyeToAt = (at - eye).normalized();
	const QVector3D up = QVector3D::crossProduct(x, eyeToAt).normalized();

	return std::make_tuple(eye, at, up);
}

cv::Vec3f reconstructPointFromViews(
	const std::vector<cv::Mat1f>& homographies,
	const std::vector<cv::Vec2f>& points
)
{
	assert(homographies.size() == points.size());
	
	const auto numberViews = int(homographies.size());

	// For the linear system Ax=b to solve
	cv::Mat1f A(3 * numberViews, 3 + numberViews, 0.0f);
	cv::Mat1f b(6, 1, 0.0f);
	for (int v = 0; v < numberViews; v++)
	{
		// Copy the 3x3 rotation matrix from this view, to the left of the matrix A
		const auto& R = homographies[v](cv::Range(0, 3), cv::Range(0, 3));
		A(cv::Range(3 * v, 3 * (v + 1)), cv::Range(0, 3)) = R * 1.f;

		b.at<float>(3 * v) = -homographies[v].at<float>(0, 3);
		b.at<float>(3 * v + 1) = -homographies[v].at<float>(1, 3);
		b.at<float>(3 * v + 2) = -homographies[v].at<float>(2, 3);

		// Copy the coordinates of the corresponding point in the matrix A
		A.at<float>(3 * v    , 3 + v) = -points[v][0];
		A.at<float>(3 * v + 1, 3 + v) = -points[v][1];
		A.at<float>(3 * v + 2, 3 + v) = -1.f;
	}

	// Solve for linear least squares
	cv::Mat1f x;
	cv::solve(A, b, x, cv::DECOMP_SVD);

	// TODO: implement non-linear optimization to refine the result

	// Convert to vector
	return {
		x.at<float>(0),
		x.at<float>(1),
		x.at<float>(2)
	};
}

cv::Vec3f lookAtPoint(const cv::Mat1f& homography, const cv::Mat1f& cameraMatrix)
{
	assert(homography.rows == 3);
	assert(homography.cols == 4);

	// For the linear system Ax=b to solve
	cv::Mat1f A(3, 3, 0.0f);
	A.at<float>(0, 0) = homography.at<float>(0, 0);
	A.at<float>(0, 1) = homography.at<float>(0, 1);
	A.at<float>(1, 0) = homography.at<float>(1, 0);
	A.at<float>(1, 1) = homography.at<float>(1, 1);
	A.at<float>(2, 0) = homography.at<float>(2, 0);
	A.at<float>(2, 1) = homography.at<float>(2, 1);

	A.at<float>(0, 2) = -cameraMatrix.at<float>(0, 2);
	A.at<float>(1, 2) = -cameraMatrix.at<float>(1, 2);
	A.at<float>(2, 2) = -1.0f;

	cv::Mat1f b(3, 1, 0.0f);
	b.at<float>(0) = -homography.at<float>(0, 3);
	b.at<float>(1) = -homography.at<float>(1, 3);
	b.at<float>(2) = -homography.at<float>(2, 3);

	// Solve for linear least squares
	cv::Mat1f x;
	cv::solve(A, b, x, cv::DECOMP_SVD);

	// Convert to vector
	return {
		x.at<float>(0),
		x.at<float>(1),
		0.0f
	};
}