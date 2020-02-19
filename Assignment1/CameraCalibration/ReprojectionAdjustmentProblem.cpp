#include "ReprojectionAdjustmentProblem.h"

#include <dlib/optimization.h>

#include "Utils.h"

using namespace dlib;

typedef matrix<float, 3, 1> parameterVector;

parameterVector vectorToParameters(const cv::Vec3f& vector)
{
	parameterVector x;
	x(0) = vector[0];
	x(1) = vector[1];
	x(2) = vector[2];

	return x;
}

cv::Vec3f parametersToVector(const parameterVector& parameters)
{
	return {
		parameters(0),
		parameters(1),
		parameters(2)
	};
}

// Takes an input, run it through the model and compare it to the output
float residual(const std::pair<cv::Mat1f, cv::Vec2f>& data, const parameterVector& parameters)
{
	// Reconstruct the vector with the parameters
	const auto point = parametersToVector(parameters);
	const cv::Vec4f homogeneousPoint(point[0], point[1], point[2], 1.0f);

	// Get points from data
	const auto& homography = data.first;
	const auto& trueProjectedPoint = data.second;

	const auto projectedPoint = projectPoint(homography, homogeneousPoint);
	
	const auto error = cv::norm(projectedPoint, trueProjectedPoint, cv::NORM_L2SQR);

	if (isnan(error))
	{
		std::cout << trans(parameters) << std::endl;
		std::cout << projectedPoint << " " << trueProjectedPoint << " " << error << std::endl;
		std::cout << std::endl << std::endl;
	}

	return error;
}

cv::Vec3f reprojectionAdjustment(
	const std::vector<cv::Mat1f>& homographies,
	const std::vector<cv::Vec2f>& points,
	const cv::Vec3f& initialGuess)
{
	assert(homographies.size() == points.size());

	std::vector<std::pair<cv::Mat1f, cv::Vec2f>> data;
	data.reserve(homographies.size() + points.size());
	for (unsigned int i = 0; i < homographies.size(); i++)
	{
		data.emplace_back(homographies[i], points[i]);
	}

	auto x = vectorToParameters(initialGuess);

	// Optimization
	solve_least_squares(objective_delta_stop_strategy(1e-8),
		                residual,
		                derivative(residual, 1e-6),
		                data,
		                x);

	return parametersToVector(x);
}
