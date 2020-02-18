#include "Keypoints.h"

#include <fstream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>

void Keypoints::clear()
{
	m_points.clear();
}

int Keypoints::size() const
{
	return m_points.size();
}

bool Keypoints::hasPoint(int keypoint, int image) const
{
	assert(keypoint >= 0);
	assert(keypoint < m_points.size());

	for (const auto& point : m_points[keypoint])
	{
		if (point.first == image)
		{
			return true;
		}
	}

	return false;
}

cv::Vec2f Keypoints::getPoint(int keypoint, int image) const
{
	assert(keypoint >= 0);
	assert(keypoint < m_points.size());
	
	cv::Vec2f result;

	for (const auto& point : m_points[keypoint])
	{
		if (point.first == image)
		{
			return point.second;
		}
	}
	
	return cv::Vec2f(-1.f, -1.f);
}

const std::vector<Keypoints::ImageKeypoint>& Keypoints::getPointsInImages(int i) const
{
	assert(i >= 0);
	assert(i < m_points.size());
	
	return m_points[i];
}

std::vector<cv::Vec2f> Keypoints::getPointInImage(int i) const
{
	std::vector<cv::Vec2f> points;

	for (const auto& image : m_points)
	{
		for (const auto& point : image)
		{
			if (point.first == i)
			{
				points.push_back(point.second);
			}
		}
	}
	
	return points;
}

bool Keypoints::load(const std::string& filename)
{
	// Read file
	std::ifstream file(filename);

	if (!file.is_open())
	{
		return false;
	}

	// Number of keypoints in the file
	int numberKeypoints;
	file >> numberKeypoints;

	for (int i = 0; i < numberKeypoints; i++)
	{
		int numberOfImages;
		file >> numberOfImages;

		std::vector<ImageKeypoint> pointsInImage;
		for (int j = 0; j < numberOfImages; j++)
		{
			int image;
			cv::Vec2f point;

			file >> image >> point[0] >> point[1];

			pointsInImage.emplace_back(image, point);
		}

		m_points.push_back(pointsInImage);
	}

	file.close();

	return true;
}

void showOrbKeypoints(const cv::Mat& image, const std::string& outputFilename)
{
	// Convert image to gray
	cv::Mat imageGray;
	cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);

	// Variables to store keypoints and descriptors
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;

	// Find key-points
	cv::Ptr<cv::Feature2D> orb = cv::ORB::create();
	orb->detectAndCompute(imageGray, cv::Mat(), keypoints, descriptors);

	cv::Mat output;
	cv::drawKeypoints(image, keypoints, output, cv::Scalar(0, 255, 0));
	cv::imwrite(outputFilename, output);
}
