#pragma once

#include <opencv2/core/core.hpp>

#include <dlib/matrix/matrix.h>

class HomographyProblem
{
public:
	/**
	 * \brief Column vector type used by dlib
	 */
	using ColumnVector = dlib::matrix<double, 0, 1>;

	HomographyProblem(
		const std::vector<cv::Vec3f>& objectPoints,
		const std::vector<cv::Vec2f>& imagePoints
	);

	double operator()(const ColumnVector& parameters) const;

	static cv::Mat1f convertVectorToMatrix(const ColumnVector& parameters);

	static ColumnVector convertMatrixToVector(const cv::Mat1f& matrix);

private:
	const std::vector<cv::Vec3f> m_objectPoints;
	const std::vector<cv::Vec2f> m_imagePoints;
};

cv::Mat1f refineHomography(
	const std::vector<cv::Vec3f>& objectPoints,
	const std::vector<cv::Vec2f>& imagePoints,
	const cv::Mat1f& startingHomography
);
