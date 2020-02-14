#include "Assignment2.h"

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "Reconstruction.h"

void Assignment2()
{
	const cv::Size chessboardSize(6, 9);
	const auto chessboardSquareSide = 0.0228f;
	
	const std::string directory = "Images/3DScene/";
	
	// Image files
	const std::vector<std::string> imageFiles = {
		"IMG_20200213_165939",
		"IMG_20200213_165943",
		"IMG_20200213_165946",
		"IMG_20200213_165951",
		"IMG_20200213_165954",
		"IMG_20200213_165956",
		"IMG_20200213_170000",
		"IMG_20200213_170007",
		"IMG_20200213_170010",
		"IMG_20200213_170014",
		"IMG_20200213_170017",
		"IMG_20200213_170020",
		"IMG_20200213_170028",
		"IMG_20200213_170032",
		"IMG_20200213_170035",
		"IMG_20200213_170038"
	};

	// Correspondences between 3D points and 2D points in each view 
	std::vector<std::vector<cv::Vec3f>> objectPoints;
	std::vector<std::vector<cv::Vec2f>> imagePoints;
	
	// Load images
	std::cout << "Reading images and extracting calibration patterns" << std::endl;
	std::vector<cv::Mat> imagesRaw;
	imagesRaw.reserve(imageFiles.size());
	for (const auto& file: imageFiles)
	{
		const auto image = cv::imread(directory + file + ".jpg");
		imagesRaw.push_back(image);

		findCorners(image,
			        objectPoints, 
			        imagePoints, 
			        directory + "keypoints/" + file + ".jpg",
					chessboardSize,
					chessboardSquareSide);
	}

	// Calibration cameras
	std::cout << "Calibrating cameras" << std::endl;
	cv::Mat cameraMatrix;
	cv::Mat distCoeffs;
	std::vector<cv::Mat> rvecs;
	std::vector<cv::Mat> tvecs;
	const auto error = cv::calibrateCamera(objectPoints,
		                                   imagePoints,
		                                   imagesRaw.front().size(),
		                                   cameraMatrix,
		                                   distCoeffs,
		                                   rvecs,
		                                   tvecs);

	std::cout << "Calibration error = " << error << std::endl;

	std::cout << "Camera matrix = " << std::endl << " " << cameraMatrix << std::endl;
	std::cout << "Distortion coefficients = " << distCoeffs << std::endl;

	for (unsigned int i = 0; i < objectPoints.size(); i++)
	{
		std::cout << "View " << i << std::endl;

		// Rotation vector
		std::cout << "  rvec = " << rvecs[i].t() << std::endl;

		// Translation vector
		std::cout << "  tvec = " << tvecs[i].t() << std::endl;
	}

	// Compute undistorted images
	std::cout << "Undistorting images" << std::endl;
	std::vector<cv::Mat> images(imagesRaw.size());
	for (unsigned int i = 0; i < imagesRaw.size(); i++)
	{
		cv::undistort(imagesRaw[i], images[i], cameraMatrix, distCoeffs);
		cv::imwrite(directory + "undistorted/" + imageFiles[i] + ".jpg", images[i]);
	}
	
	// Since images are undistorted, we can now set the distortion coefficients to zero
	distCoeffs.setTo(0.0f);

	std::cout << "Reprojecting undistorted images" << std::endl;
	for (unsigned int i = 0; i < images.size(); i++)
	{
		drawProjectedCorners(images[i],
			                 objectPoints[i],
			                 cameraMatrix,
			                 distCoeffs,
			                 rvecs[i],
			                 tvecs[i],
			                 directory + "reprojections/" + imageFiles[i] + ".jpg");
	}
}