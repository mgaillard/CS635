#include "Keypoints.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>

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
