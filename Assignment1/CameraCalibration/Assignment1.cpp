#include "Assignment1.h"

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "Utils.h"
#include "Reconstruction.h"
#include "HomographyProblem.h"
#include "BundleAdjustmentProblem.h"

cv::Mat findHomography(
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

	// Create the matrix L to solve for an initial value of the homography
	cv::Mat1f L(2 * objectPoints.size(), 9, 0.0f);

	for (int i = 0; i < objectPoints.size(); i++)
	{
		L.at<float>(2 * i, 0) = objectPoints[i][0];
		L.at<float>(2 * i, 1) = objectPoints[i][1];
		L.at<float>(2 * i, 2) = 1.0f;
		L.at<float>(2 * i, 6) = -imagePoints[i][0] * objectPoints[i][0];
		L.at<float>(2 * i, 7) = -imagePoints[i][0] * objectPoints[i][1];
		L.at<float>(2 * i, 8) = -imagePoints[i][0];

		L.at<float>(2 * i + 1, 3) = objectPoints[i][0];
		L.at<float>(2 * i + 1, 4) = objectPoints[i][1];
		L.at<float>(2 * i + 1, 5) = 1.0f;
		L.at<float>(2 * i + 1, 6) = -imagePoints[i][1] * objectPoints[i][0];
		L.at<float>(2 * i + 1, 7) = -imagePoints[i][1] * objectPoints[i][1];
		L.at<float>(2 * i + 1, 8) = -imagePoints[i][1];
	}

	// We are looking at the right singular vector of L associated to the smallest singular value
	cv::Mat w, u, vt;
	cv::SVD::compute(L, w, u, vt, cv::SVD::FULL_UV);

	// Last row of vt contains the initial guess for H
	auto H = vt.row(8).reshape(0, 3);

	// Refine H based on the least square minimization
	H = refineHomography(objectPoints, imagePoints, H);

	// Normalize the H matrix
	H /= H.at<float>(2, 2);

	return H;
}

cv::Mat findHomographyOpenCV(
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

	// As a reference, this is how we could get the homography directly from OpenCV
	cv::Mat H = cv::findHomography(objectPointsPlanar, imagePoints);

	// Normalize the H matrix
	H /= H.at<float>(2, 2);

	return H;
}

cv::Mat1f computeV(const cv::Mat1f& H, int i, int j)
{
	assert(H.rows == 3);
	assert(H.cols == 3);

	// i and j are indexed from 0
	i -= 1;
	j -= 1;

	cv::Mat1f v(6, 1, 0.0f);

	v.at<float>(0) = H.at<float>(0, i) * H.at<float>(0, j);
	v.at<float>(1) = H.at<float>(0, i) * H.at<float>(1, j) + H.at<float>(1, i) * H.at<float>(0, j);
	v.at<float>(2) = H.at<float>(1, i) * H.at<float>(1, j);
	v.at<float>(3) = H.at<float>(2, i) * H.at<float>(0, j) + H.at<float>(0, i) * H.at<float>(2, j);
	v.at<float>(4) = H.at<float>(2, i) * H.at<float>(1, j) + H.at<float>(1, i) * H.at<float>(2, j);
	v.at<float>(5) = H.at<float>(2, i) * H.at<float>(2, j);

	return v;
}

cv::Mat1f formSymmetricMat(const cv::Mat1f& b)
{
	assert(b.rows == 6);
	assert(b.cols == 1);

	cv::Mat1f B(3, 3, 0.0f);

	// First row
	B.at<float>(0, 0) = b.at<float>(0);
	B.at<float>(0, 1) = b.at<float>(1);
	B.at<float>(0, 2) = b.at<float>(3);

	// Second row
	B.at<float>(1, 0) = b.at<float>(1);
	B.at<float>(1, 1) = b.at<float>(2);
	B.at<float>(1, 2) = b.at<float>(4);

	// Third row
	B.at<float>(2, 0) = b.at<float>(3);
	B.at<float>(2, 1) = b.at<float>(4);
	B.at<float>(2, 2) = b.at<float>(5);

	return B;
}

cv::Mat1f solveCameraCalibration(const cv::Mat1f& homography1, const cv::Mat1f& homography2)
{
	assert(homography1.rows == 3);
	assert(homography1.cols == 3);

	assert(homography2.rows == 3);
	assert(homography2.cols == 3);

	const std::array<const cv::Mat1f*, 2> homographies = { {&homography1, &homography2} };

	// We try to find the matrix B that solves the equation (3) and (4) for each view
	// Two homographies of the same pattern means 4 equations
	cv::Mat1f V(5, 6, 0.0f);
	for (unsigned int i = 0; i < homographies.size(); i++)
	{
		const auto v11 = computeV(*homographies[i], 1, 1);
		const auto v12 = computeV(*homographies[i], 1, 2);
		const auto v22 = computeV(*homographies[i], 2, 2);

		V.row(2 * i) = v12.t();
		V.row(2 * i + 1) = (v11 - v22).t();
	}

	// Since we have only two views, impose the skewless constraint gamma = 0
	// Row 4 of the V matrix is [0, 1, 0, 0, 0, 0]
	V.at<float>(4, 1) = 1.0f;

	// We are looking at the right singular vector of V associated to the smallest singular value
	cv::Mat w, u, vt;
	cv::SVD::compute(V, w, u, vt, cv::SVD::FULL_UV);

	// Last row of vt contains the initial guess for b
	const auto b = vt.row(5).reshape(0, 6);
	const auto B = formSymmetricMat(b);

	// Display error values of the equations for debugging purpose
	/*
	std::cout << "b = " << std::endl << b << std::endl;
	for (unsigned int i = 0; i < homographies.size(); i++)
	{
		const auto eq3 = homographies[i]->col(0).t() * B * homographies[i]->col(1);
		const auto eq4 = homographies[i]->col(0).t() * B * homographies[i]->col(0)
					   - homographies[i]->col(1).t() * B * homographies[i]->col(1);

		std::cout << "Eq 3: " << eq3 << std::endl;
		std::cout << "Eq 4: " << eq4 << std::endl;
	}
	std::cout << "V*b = " << std::endl << V*b << std::endl;
	*/

	const auto b11 = b.at<float>(0);
	const auto b12 = b.at<float>(1);
	const auto b22 = b.at<float>(2);
	const auto b13 = b.at<float>(3);
	const auto b23 = b.at<float>(4);
	const auto b33 = b.at<float>(5);

	const auto v0 = (b12 * b13 - b11 * b23) / (b11 * b22 - b12 * b12);
	const auto lambda = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11;
	const auto alpha = std::sqrt(lambda / b11);
	const auto beta = std::sqrt(lambda * b11 / (b11 * b22 - b12 * b12));
	const auto gamma = -b12 * alpha * alpha * beta / lambda;
	const auto u0 = gamma * v0 / beta - b13 * alpha * alpha / lambda;

	cv::Mat1f A(3, 3, 0.0f);
	A.at<float>(0, 0) = alpha;
	A.at<float>(0, 2) = u0;
	A.at<float>(1, 1) = beta;
	A.at<float>(1, 2) = v0;
	A.at<float>(2, 2) = 1.0f;

	return A;
}

std::pair<cv::Mat1f, cv::Mat1f> computeExtrinsicParameters(const cv::Mat1f& homography, const cv::Mat1f& A)
{
	assert(homography.rows == 3);
	assert(homography.cols == 3);

	assert(A.rows == 3);
	assert(A.cols == 3);

	// Inverse the intrinsic matrix
	const cv::Mat1f invA = A.inv();

	const cv::Mat1f h1 = homography.col(0);
	const cv::Mat1f h2 = homography.col(1);
	const cv::Mat1f h3 = homography.col(2);

	// Rotation matrix
	cv::Mat1f R(3, 3, 0.0f);
	R.col(0) = invA * h1 / cv::norm(invA * h1);
	R.col(1) = invA * h2 / cv::norm(invA * h2);
	R.col(2) = 1.0f * R.col(0).cross(R.col(1));

	// Translation
	const cv::Mat1f t = invA * h3 / cv::norm(invA * h1);

	return std::make_pair(R, t);
}

void Assignment1()
{
	const auto imageFront = cv::imread("Images/front.jpg");
	const auto imageLeft = cv::imread("Images/left.jpg");

	// Correspondences between 3D points and 2D points in each view 
	std::vector<std::vector<cv::Vec3f>> objectPoints;
	std::vector<std::vector<cv::Vec2f>> imagePoints;

	// Space between two corners (4 in = 0.1016 m)
	const auto squareSize = 0.1016f;
	findCorners(imageFront, objectPoints, imagePoints, "Images/front_keypoints.jpg", cv::Size(11, 17), squareSize);
	findCorners(imageLeft, objectPoints, imagePoints, "Images/left_keypoints.jpg", cv::Size(11, 17), squareSize);
	// In the left image, OpenCV finds the chess board upside down, so we reverse the order of image points
	std::reverse(imagePoints[1].begin(), imagePoints[1].end());


	std::cout << "Find homographies:" << std::endl
		<< "==================" << std::endl;

	const auto H1 = findHomography(objectPoints[0], imagePoints[0]);
	const auto H2 = findHomography(objectPoints[1], imagePoints[1]);
	const auto H1OpenCV = findHomographyOpenCV(objectPoints[0], imagePoints[0]);
	const auto H2OpenCV = findHomographyOpenCV(objectPoints[1], imagePoints[1]);

	std::cout << "View 1 re-projection error (in px) = " << computeAvgReProjectionError(objectPoints[0], imagePoints[0], H1)
		<< "\t OpenCV reference = " << computeAvgReProjectionError(objectPoints[0], imagePoints[0], H1OpenCV) << std::endl;

	std::cout << "View 2 re-projection error (in px) = " << computeAvgReProjectionError(objectPoints[1], imagePoints[1], H2)
		<< "\t OpenCV reference = " << computeAvgReProjectionError(objectPoints[1], imagePoints[1], H2OpenCV) << std::endl;

	std::cout << std::endl << std::endl;

	const auto A = solveCameraCalibration(H1, H2);

	const auto extrinsicParameters1 = computeExtrinsicParameters(H1, A);
	const auto extrinsicParameters2 = computeExtrinsicParameters(H2, A);

	std::vector<cv::Mat1f> guessRvecs;
	std::vector<cv::Mat1f> guessTvecs;

	cv::Mat1f angles1;
	cv::Rodrigues(extrinsicParameters1.first, angles1);
	guessRvecs.push_back(angles1);
	guessTvecs.push_back(extrinsicParameters1.second);

	cv::Mat1f angles2;
	cv::Rodrigues(extrinsicParameters2.first, angles2);
	guessRvecs.push_back(angles2);
	guessTvecs.push_back(extrinsicParameters2.second);

	// Bundle adjustment
	cv::Mat1f cameraMatrix;
	cv::Mat1f distCoeffs;
	std::vector<cv::Mat1f> rvecs;
	std::vector<cv::Mat1f> tvecs;
	std::tie(cameraMatrix, distCoeffs, rvecs, tvecs) = bundleAdjustment(objectPoints, imagePoints, A, guessRvecs, guessTvecs);

	std::cout << "Camera calibration with distortion using OpenCV (for reference)" << std::endl << std::endl;
	cv::Mat cameraMatrixOpenCV;
	cv::Mat distCoeffsOpenCV;
	std::vector<cv::Mat> rvecsOpenCV;
	std::vector<cv::Mat> tvecsOpenCV;
	const auto error = cv::calibrateCamera(objectPoints,
		imagePoints,
		imageFront.size(),
		cameraMatrixOpenCV,
		distCoeffsOpenCV,
		rvecsOpenCV,
		tvecsOpenCV);

	std::cout << "Results of camera calibration" << std::endl
		<< "=============================" << std::endl;

	// Camera matrix M
	std::cout << "Camera matrix = " << std::endl << " " << cameraMatrix << std::endl;
	std::cout << "OpenCV reference = " << std::endl << " " << cameraMatrixOpenCV << std::endl << std::endl;

	// For a Google Pixel 3, the sensor is 5.76 mm by 4.29 mm
	const cv::Size2f sensorSize(5.76f, 4.29f);
	const auto focalLength = focalLengthInMm(cameraMatrix, imageFront.size(), sensorSize);
	const auto focalLengthOpenCV = focalLengthInMm(cameraMatrixOpenCV, imageFront.size(), sensorSize);
	std::cout << "Focal length (in mm): width = " << focalLength.first
		      << " height = " << focalLength.second
		      << "\t OpenCV reference: width = " << focalLengthOpenCV.first
		      << " height = " << focalLengthOpenCV.second << std::endl << std::endl;

	// Distortion
	std::cout << "Distortion coefficients = " << distCoeffs << std::endl;
	std::cout << "       OpenCV reference = " << distCoeffsOpenCV << std::endl << std::endl;

	for (unsigned int i = 0; i < objectPoints.size(); i++)
	{
		std::cout << "View " << i << std::endl;

		// Rotation vector
		std::cout << "  rvec = " << rvecs[i].t() << std::endl;
		std::cout << "OpenCV = " << rvecsOpenCV[i].t() << std::endl << std::endl;
		// cv::Mat1f rvecsMatrix;
		// cv::Rodrigues(rvecs[i], rvecsMatrix);
		// std::cout << "matrix = " << std::endl << " " << rotationX180(rvecsMatrix).t() << std::endl << std::endl;
		// std::cout << "euler = " << std::endl << " " << rotationMatrixToEulerAnglesDeg(rotationX180(rvecsMatrix).t()) << std::endl << std::endl;

		// Translation vector
		std::cout << "  tvec = " << tvecs[i].t() << std::endl;
		std::cout << "OpenCV = " << tvecsOpenCV[i].t() << std::endl << std::endl;
	}

	// Manually compute the re-projection error
	const auto rmsError = computeRMSReProjectionError(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvecs, tvecs);
	const auto rmsErrorOpenCV = computeRMSReProjectionError(objectPoints, imagePoints, cameraMatrixOpenCV, distCoeffsOpenCV, rvecsOpenCV, tvecsOpenCV);
	const auto avgError = computeAvgReProjectionError(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvecs, tvecs);
	const auto avgErrorOpenCV = computeAvgReProjectionError(objectPoints, imagePoints, cameraMatrixOpenCV, distCoeffsOpenCV, rvecsOpenCV, tvecsOpenCV);

	std::cout << "RMS re-projection error = " << rmsError << std::endl;
	std::cout << "       OpenCV reference = " << rmsErrorOpenCV << std::endl;
	std::cout << "AVG re-projection error = " << avgError << std::endl;
	std::cout << "       OpenCV reference = " << avgErrorOpenCV << std::endl << std::endl;

	drawProjectedCorners(imageFront, objectPoints[0], cameraMatrix, distCoeffs, rvecs[0], tvecs[0], "Images/front_projection.jpg");
	drawProjectedCorners(imageLeft, objectPoints[1], cameraMatrix, distCoeffs, rvecs[1], tvecs[1], "Images/left_projection.jpg");

	std::cout << "Camera pose:" << std::endl
		<< "============" << std::endl;
	cameraPose(rvecs[0], tvecs[0]);
	cameraPose(rvecs[1], tvecs[1]);
}
