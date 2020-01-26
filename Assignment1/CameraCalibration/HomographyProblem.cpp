#include "HomographyProblem.h"

#include <dlib/optimization.h>

#include "Utils.h"

using namespace dlib;

HomographyProblem::HomographyProblem(
	const std::vector<cv::Vec3f>& objectPoints,
	const std::vector<cv::Vec2f>& imagePoints) :
	m_objectPoints(objectPoints),
	m_imagePoints(imagePoints)
{
	
}

double HomographyProblem::operator()(const ColumnVector& parameters) const
{
	const auto homography = convertVectorToMatrix(parameters);
	
	// Compute error between object projected points and image points

	double meanError = 0.0f;

	for (unsigned int i = 0; i < m_objectPoints.size(); i++)
	{
		const auto projectedPoint = projectPoint(homography, m_objectPoints[i]);

		meanError += cv::norm(m_imagePoints[i], projectedPoint, cv::NORM_L2SQR);
	}

	return meanError / m_objectPoints.size();
}

cv::Mat1f HomographyProblem::convertVectorToMatrix(const ColumnVector& parameters)
{
	cv::Mat1f matrix(3, 3, 0.0f);

	for (int i = 0; i < 9; i++)
	{
		matrix.at<float>(i) = parameters(i);
	}
	
	return matrix;
}

HomographyProblem::ColumnVector HomographyProblem::convertMatrixToVector(const cv::Mat1f& matrix)
{
	ColumnVector parameters(9);

	for (int i = 0; i < 9; i++)
	{
		parameters(i) = matrix.at<float>(i);
	}
	
	return parameters;
}

cv::Mat1f refineHomography(
	const std::vector<cv::Vec3f>& objectPoints,
	const std::vector<cv::Vec2f>& imagePoints,
	const cv::Mat1f& startingHomography)
{
	const HomographyProblem problem(objectPoints, imagePoints);
	
	auto parameters = HomographyProblem::convertMatrixToVector(startingHomography);

	// std::cout << "Initial cost: " << problem(parameters) << std::endl;

	find_min_using_approximate_derivatives(bfgs_search_strategy(),
		                                   objective_delta_stop_strategy(1e-8, 100),
		                                   problem,
		                                   parameters,
		                                   0);

	// std::cout << "Final cost: " << problem(parameters) << std::endl;

	return HomographyProblem::convertVectorToMatrix(parameters);
}
