#include "MainWindow.h"

#include <iostream>

#include <QTimer>
#include <QtMath>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>


#include "MeshObject.h"
#include "PhysicalCamera.h"
#include "Reconstruction.h"
#include "Utils.h"

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
{
	setupUi();

	// Setup the scene, once all widgets are loaded
	QTimer::singleShot(0, this, &MainWindow::setupScene);
}

void MainWindow::setupScene()
{
	setupPattern();
	setupCameras();
}

void MainWindow::selectCameraClicked(int camera)
{
	if (camera >= 0 && camera < m_cameras.size())
	{
		// Set the camera in the viewer
		const OrbitCamera orbitCamera(m_cameras[camera]);
		m_ui.viewerWidget->setCamera(orbitCamera);
	}
}

void MainWindow::setupUi()
{
	m_ui.setupUi(this);

	connect(m_ui.actionSelect_camera_1, &QAction::triggered, [this]() { selectCameraClicked(0); });
	connect(m_ui.actionSelect_camera_2, &QAction::triggered, [this]() { selectCameraClicked(1); });
	connect(m_ui.actionSelect_camera_3, &QAction::triggered, [this]() { selectCameraClicked(2); });
	connect(m_ui.actionSelect_camera_4, &QAction::triggered, [this]() { selectCameraClicked(3); });
	connect(m_ui.actionSelect_camera_5, &QAction::triggered, [this]() { selectCameraClicked(4); });
	connect(m_ui.actionSelect_camera_6, &QAction::triggered, [this]() { selectCameraClicked(5); });
	connect(m_ui.actionSelect_camera_7, &QAction::triggered, [this]() { selectCameraClicked(6); });
	connect(m_ui.actionSelect_camera_8, &QAction::triggered, [this]() { selectCameraClicked(7); });
	connect(m_ui.actionSelect_camera_9, &QAction::triggered, [this]() { selectCameraClicked(8); });
	connect(m_ui.actionSelect_camera_10, &QAction::triggered, [this]() { selectCameraClicked(9); });
	connect(m_ui.actionSelect_camera_11, &QAction::triggered, [this]() { selectCameraClicked(10); });
	connect(m_ui.actionSelect_camera_12, &QAction::triggered, [this]() { selectCameraClicked(11); });
	connect(m_ui.actionSelect_camera_13, &QAction::triggered, [this]() { selectCameraClicked(12); });
	connect(m_ui.actionSelect_camera_14, &QAction::triggered, [this]() { selectCameraClicked(13); });
	connect(m_ui.actionSelect_camera_15, &QAction::triggered, [this]() { selectCameraClicked(14); });
	connect(m_ui.actionSelect_camera_16, &QAction::triggered, [this]() { selectCameraClicked(15); });
}

void MainWindow::setupPattern()
{
	std::vector<QVector3D> vertices = {
		{   0.0f,    0.0f, 0.0f},
		{0.0228f,    0.0f, 0.0f},
		{   0.0f, 0.0228f, 0.0f},
		{0.0228f, 0.0228f, 0.0f}
	};
	
	std::vector<std::tuple<int, int, int>> faces = {
		std::make_tuple(0, 1, 2),
		std::make_tuple(1, 3, 2)
	};

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			QMatrix4x4 meshWorldMatrix;

			meshWorldMatrix.translate(
				2 * i * 0.0228f,
				2 * j * 0.0228f,
				0.0f
			);
			
			auto mesh = std::make_unique<MeshObject>(meshWorldMatrix, vertices, faces);
			m_ui.viewerWidget->addObject(std::move(mesh));
		}
	}
}

void MainWindow::setupCameras()
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

	const auto imageWidth = imagesRaw.front().cols;
	const auto imageHeight = imagesRaw.front().rows;

	// Calibration cameras
	cv::Mat1f cameraMatrix;
	cv::Mat1f distCoeffs;
	std::vector<cv::Mat> rvecs;
	std::vector<cv::Mat> tvecs;
	const auto error = cv::calibrateCamera(objectPoints,
		                                   imagePoints,
		                                   imagesRaw.front().size(),
		                                   cameraMatrix,
		                                   distCoeffs,
		                                   rvecs,
		                                   tvecs);
	
	std::vector<cv::Mat> images(imagesRaw.size());
	for (unsigned int i = 0; i < imagesRaw.size(); i++)
	{
		cv::undistort(imagesRaw[i], images[i], cameraMatrix, distCoeffs);
		cv::imwrite(directory + "undistorted/" + imageFiles[i] + ".jpg", images[i]);
	}
	
	// Since images are undistorted, we can now set the distortion coefficients to zero
	distCoeffs.setTo(0.0f);

	// Compute aspect ratio from images
	const auto aspectRatio = float(imageWidth) / float(imageHeight);

	// For a Google Pixel 3, the sensor is 5.76 mm by 4.29 mm
	const cv::Size2f sensorSize(5.76f, 4.29f);
	const auto focalLength = focalLengthInMm(cameraMatrix, imagesRaw.front().size(), sensorSize);
	const auto fovy = qRadiansToDegrees(2.0f * std::atan(sensorSize.height / (2.0f * focalLength.second)));

	// Shift image according to the center of the optical axis in the image
	const float imageCenterWidth = cameraMatrix.at<float>(0, 2);
	const float imageCenterHeight = cameraMatrix.at<float>(1, 2);
	const float imageShiftWidth = imageWidth / 2 - imageCenterWidth;
	const float imageShiftHeight = imageHeight / 2 - imageCenterHeight;

	// Pose of cameras in world coordinates
	for (unsigned int i = 0; i < imagesRaw.size(); i++)
	{
		QVector3D eye, at, up;
		std::tie(eye, at, up) = cameraEyeAtUpFromPose(cameraMatrix, rvecs[i], tvecs[i]);

		Camera camera(
			eye,
			at,
			up,
			fovy,
			aspectRatio,
			0.001f,
			10.0f
		);

		// Add the camera to the list of cameras
		m_cameras.push_back(camera);
				
		// Shift the image according to the optical center
		const auto shiftedImage = translateImage(images[i], imageShiftWidth, imageShiftHeight);

		// Add the physical camera to the view
		auto physicalCamera = std::make_unique<PhysicalCamera>(
			camera,
			0.05f,
			convertToQtImage(shiftedImage)
		);
		m_ui.viewerWidget->addObject(std::move(physicalCamera));
	}
}
