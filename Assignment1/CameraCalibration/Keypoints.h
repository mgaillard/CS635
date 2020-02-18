#pragma once

#include <string>

#include <opencv2/core/core.hpp>

class Keypoints
{
public:
	using ImageKeypoint = std::pair<int, cv::Vec2f>;
	
	Keypoints() = default;

	void clear();

	int size() const;

	// Add a keypoints with his coordinates in at least two images
	void add(std::vector<ImageKeypoint> pointsInImages);

	// Return the coordinates of the keypoints in images
	const std::vector<ImageKeypoint>& getPointsInImages(int i) const;

	// TODO: get all the keypoints in one image

	bool load(const std::string& filename);
	
private:
	// Index keypoint and then images
	std::vector<std::vector<ImageKeypoint>> m_points;
};

void showOrbKeypoints(const cv::Mat& image, const std::string& outputFilename);
