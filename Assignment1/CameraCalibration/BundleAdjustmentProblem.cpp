#include "BundleAdjustmentProblem.h"

#include <opencv2/calib3d.hpp>
#include <dlib/optimization.h>

#include "Utils.h"

using namespace dlib;

typedef matrix<float, 16, 1> parameterVector;

parameterVector matricesToParameters(
	const cv::Mat1f& cameraMatrix,
	const std::vector<cv::Mat1f>& rvecs,
	const std::vector<cv::Mat1f>& tvecs)
{
	parameterVector x;
	x(0) = cameraMatrix.at<float>(0, 0) / 1000.0f; // Alpha
	x(1) = cameraMatrix.at<float>(1, 1) / 1000.0f; // Beta
	x(2) = cameraMatrix.at<float>(0, 2) / 1000.0f; // u0
	x(3) = cameraMatrix.at<float>(1, 2) / 1000.0f; // v0
	x(4) = rvecs[0].at<float>(0); // view 1 r1
	x(5) = rvecs[0].at<float>(1); // view 1 r2
	x(6) = rvecs[0].at<float>(2); // view 1 r3
	x(7) = tvecs[0].at<float>(0); // view 1 t1
	x(8) = tvecs[0].at<float>(1); // view 1 t2
	x(9) = tvecs[0].at<float>(2); // view 1 t3
	x(10) = rvecs[1].at<float>(0); // view 2 r1
	x(11) = rvecs[1].at<float>(1); // view 2 r2
	x(12) = rvecs[1].at<float>(2); // view 2 r3
	x(13) = tvecs[1].at<float>(0); // view 2 t1
	x(14) = tvecs[1].at<float>(1); // view 2 t2
	x(15) = tvecs[1].at<float>(2); // view 2 t3

	return x;
}

std::tuple<cv::Mat1f, std::vector<cv::Mat1f>, std::vector<cv::Mat1f>>
parametersToMatrices(const parameterVector& parameters)
{
	cv::Mat1f cameraMatrix(3, 3, 0.0f);
	cv::Mat1f r1(3, 1, 0.0f);
	cv::Mat1f t1(3, 1, 0.0f);
	cv::Mat1f r2(3, 1, 0.0f);
	cv::Mat1f t2(3, 1, 0.0f);

	cameraMatrix.at<float>(0, 0) = 1000.0f * parameters(0); // Alpha
	cameraMatrix.at<float>(0, 2) = 1000.0f * parameters(2); // Beta
	cameraMatrix.at<float>(1, 1) = 1000.0f * parameters(1); // u0
	cameraMatrix.at<float>(1, 2) = 1000.0f * parameters(3); // v0
	cameraMatrix.at<float>(2, 2) = 1.0f;
	r1.at<float>(0) = parameters(4); // view 1 r1
	r1.at<float>(1) = parameters(5); // view 1 r2
	r1.at<float>(2) = parameters(6); // view 1 r3
	t1.at<float>(0) = parameters(7); // view 1 t1
	t1.at<float>(1) = parameters(8); // view 1 t2
	t1.at<float>(2) = parameters(9); // view 1 t3
	r2.at<float>(0) = parameters(10); // view 2 r1
	r2.at<float>(1) = parameters(11); // view 2 r2
	r2.at<float>(2) = parameters(12); // view 2 r3
	t2.at<float>(0) = parameters(13); // view 2 t1
	t2.at<float>(1) = parameters(14); // view 2 t2
	t2.at<float>(2) = parameters(15); // view 2 t3

	std::vector<cv::Mat1f> rvecs = { r1, r2 };
	std::vector<cv::Mat1f> tvecs = { t1, t2 };

	return std::make_tuple(cameraMatrix, rvecs, tvecs);
}

// Takes an input, run it through the model and compare it to the output
float residual(const std::tuple<int, cv::Vec3f, cv::Vec2f>& data,
	           const parameterVector& parameters)
{
	// Reconstruct the matrices with the params
	cv::Mat1f cameraMatrix;
	std::vector<cv::Mat1f> rvecs;
	std::vector<cv::Mat1f> tvecs;
	std::tie(cameraMatrix, rvecs, tvecs) = parametersToMatrices(parameters);

	// Get points from data
	const auto view = std::get<0>(data);
	const std::vector<cv::Vec3f> objectPoint(1, std::get<1>(data));
	const cv::Vec2f imagePoint = std::get<2>(data);
	
	std::vector<cv::Vec2f> projectedPoints;
	cv::Mat1f distCoeffs(1, 5, 0.0f); // Empty distortion coefficients
	cv::projectPoints(objectPoint,
		              rvecs[view],
		              tvecs[view],
		              cameraMatrix,
		              distCoeffs,
		              projectedPoints);
	
	const auto error = cv::norm(imagePoint, projectedPoints.front(), cv::NORM_L2);

	if (isnan(error))
	{
		std::cout << trans(parameters) << std::endl;
		std::cout << imagePoint << " " << projectedPoints.front() << " " << error << std::endl;
		std::cout << std::endl << std::endl;
	}

	
	return error;
}

std::tuple<cv::Mat1f, std::vector<cv::Mat1f>, std::vector<cv::Mat1f>>
bundleAdjustment(
	const std::vector<std::vector<cv::Vec3f>>& objectPoints,
	const std::vector<std::vector<cv::Vec2f>>& imagePoints,
	const cv::Mat1f& cameraMatrix,
	const std::vector<cv::Mat1f>& rvecs,
	const std::vector<cv::Mat1f>& tvecs)
{
	assert(objectPoints.size() == 2);
	assert(imagePoints.size() == 2);
	assert(rvecs.size() == 2);
	assert(tvecs.size() == 2);

	std::vector<std::tuple<int, cv::Vec3f, cv::Vec2f>> data;

	data.reserve(objectPoints[0].size() + objectPoints[1].size());
	for (int i = 0; i < 2; i++)
	{
		for (unsigned int j = 0; j < objectPoints[i].size(); j++)
		{
			data.emplace_back(i, objectPoints[i][j], imagePoints[i][j]);
		}
	}

	auto x = matricesToParameters(cameraMatrix, rvecs, tvecs);
	
	solve_least_squares(objective_delta_stop_strategy(1e-7).be_verbose(),
		                residual,
		                derivative(residual, 1e-5),
		                data,
		                x);
	
	return parametersToMatrices(x);
}
