#include "MainWindow.h"

#include <iostream>

#include <QTimer>
#include <QtMath>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>


#include "Keypoints.h"
#include "MeshObject.h"
#include "PhysicalCamera.h"
#include "PointObject.h"
#include "Reconstruction.h"
#include "Triangulation.h"
#include "Utils.h"

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent),
	m_directory("Images/3DScene/"),
	m_currentCamera(0)
{
	setupUi();

	// Setup the scene, once all widgets are loaded
	QTimer::singleShot(0, this, &MainWindow::setupScene);
}

void MainWindow::setupScene()
{
	setupPattern();
	setupCameras();
	setupPhysicalCameras();
	reconstructPoints();

	// Select the first camera and reconstruct the mesh from this point of view
	selectCameraClickedWithoutReconstruction(0);
	reconstructMesh(0, false);
}

void MainWindow::selectCameraClicked(int camera)
{
	if (camera >= 0 && camera < m_cameras.size())
	{
		// Set the camera in the viewer
		selectCameraClickedWithoutReconstruction(camera);

		// Reconstruct the mesh from this view
		reconstructMesh(m_currentCamera, true);
	}
}

void MainWindow::selectCameraClickedWithoutReconstruction(int camera)
{
	if (camera >= 0 && camera < m_cameras.size())
	{
		// Set the camera in the viewer
		m_currentCamera = camera;
		const OrbitCamera orbitCamera(m_cameras[m_currentCamera]);
		m_ui.viewerWidget->setCamera(orbitCamera);
	}
}

void MainWindow::viewDoubleClicked(qreal x, qreal y)
{
	// Get the parameters of the current image, with the current camera
	const auto& currentImage = m_images[m_currentCamera];
	const auto originalImageWidth = float(currentImage.cols);
	const auto originalImageHeight = float(currentImage.rows);
	
	// Aspect ratio of the image
	const auto aspectRatio = float(originalImageWidth) / float(originalImageHeight);
	
	// Size of the viewer widget
	const auto viewerWidth = float(m_ui.viewerWidget->width());
	const auto viewerHeight = float(m_ui.viewerWidget->height());

	// The width of the image is constrained by height
	const float imageWidth = aspectRatio * viewerHeight;
	const float imageHeight = imageWidth / aspectRatio;

	// Find origin of the image
	const auto originWidth = (viewerWidth - imageWidth) / 2.f;
	const auto originHeight = (viewerHeight - imageHeight) / 2.f;
	
	// Remap the click on the image
	const auto imageX = ((x - originWidth) / imageWidth) * originalImageWidth;
	const auto imageY = ((y - originHeight) / imageHeight)* originalImageHeight;

	// Take in account the image shift
	const auto imageCenterWidth = getImageCenterX(m_cameraMatrix);
	const auto imageCenterHeight = getImageCenterY(m_cameraMatrix);
	const auto imageShiftWidth = imageCenterWidth - (originalImageWidth / 2.f);
	const auto imageShiftHeight = imageCenterHeight - (originalImageHeight / 2.f);

	m_keypointsDock->addKeypoint(m_currentCamera, imageX + imageShiftWidth, imageY + imageShiftHeight);
}

void MainWindow::setupUi()
{
	m_ui.setupUi(this);

	m_keypointsDock = new KeypointsDockWidget(this);
	addDockWidget(Qt::RightDockWidgetArea, m_keypointsDock);
	m_ui.menuBar->addAction(m_keypointsDock->toggleViewAction());

	connect(m_ui.viewerWidget, &ViewerWidget::mouseDoubleClicked, this, &MainWindow::viewDoubleClicked);
	
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

	std::vector<QVector2D> uv = {
		{0.0f, 0.0f},
		{0.0f, 0.0f},
		{0.0f, 0.0f},
		{0.0f, 0.0f}
	};
	
	std::vector<std::tuple<int, int, int>> faces = {
		std::make_tuple(0, 1, 2),
		std::make_tuple(1, 3, 2)
	};

	// Small black image
	QImage textureImage(1, 1, QImage::Format_Grayscale8);
	textureImage.fill(Qt::black);

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
			
			auto mesh = std::make_unique<MeshObject>(meshWorldMatrix, vertices, uv, faces, textureImage);
			m_ui.viewerWidget->addObject(std::move(mesh));
		}
	}
}

void MainWindow::setupCameras()
{
	const cv::Size chessboardSize(6, 9);
	const auto chessboardSquareSide = 0.0228f;
	
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
	std::vector<std::vector<cv::Vec3f>> objectPoints(imageFiles.size());
	std::vector<std::vector<cv::Vec2f>> imagePoints(imageFiles.size());
	
	// Load original images
	m_imagesRaw.clear();
	m_imagesRaw.resize(imageFiles.size());
	#pragma omp parallel for
	for (int i = 0; i < imageFiles.size(); i++)
	{
		m_imagesRaw[i] = cv::imread(m_directory + imageFiles[i] + ".jpg");

		std::vector<std::vector<cv::Vec3f>> currentObjectPoints;
		std::vector<std::vector<cv::Vec2f>> currentImagePoints;
		findCorners(m_imagesRaw[i],
			        currentObjectPoints, 
			        currentImagePoints, 
			        m_directory + "keypoints/" + imageFiles[i] + ".jpg",
					chessboardSize,
					chessboardSquareSide);

		objectPoints[i] = currentObjectPoints.front();
		imagePoints[i] = currentImagePoints.front();
	}

	// Calibration cameras
	cv::Mat1f distCoeffs;
	const auto error = cv::calibrateCamera(objectPoints,
		                                   imagePoints,
		                                   m_imagesRaw.front().size(),
		                                   m_cameraMatrix,
		                                   distCoeffs,
		                                   m_rvecs,
		                                   m_tvecs);
	
	m_images.resize(m_imagesRaw.size());
	#pragma omp parallel for
	for (int i = 0; i < m_imagesRaw.size(); i++)
	{
		cv::undistort(m_imagesRaw[i], m_images[i], m_cameraMatrix, distCoeffs);
		
		cv::imwrite(m_directory + "undistorted/" + imageFiles[i] + ".jpg", m_images[i]);
		drawProjectedCorners(m_images[i],
			                 objectPoints[i],
			                 m_cameraMatrix,
			                 distCoeffs,
			                 m_rvecs[i],
			                 m_tvecs[i],
			                 m_directory + "reprojections/" + imageFiles[i] + ".jpg");
	}
	
	// Since images are undistorted, we can now set the distortion coefficients to zero
	distCoeffs.setTo(0.0f);

	const auto imageWidth = m_imagesRaw.front().cols;
	const auto imageHeight = m_imagesRaw.front().rows;

	// Compute aspect ratio from images
	const auto aspectRatio = float(imageWidth) / float(imageHeight);

	// For a Google Pixel 3, the sensor is 5.76 mm by 4.29 mm
	const cv::Size2f sensorSize(5.76f, 4.29f);
	const auto focalLength = focalLengthInMm(m_cameraMatrix, m_imagesRaw.front().size(), sensorSize);
	const auto fovy = qRadiansToDegrees(2.0f * std::atan(sensorSize.height / (2.0f * focalLength.second)));

	// Pose of cameras in world coordinates
	for (unsigned int i = 0; i < m_imagesRaw.size(); i++)
	{
		QVector3D eye, at, up;
		std::tie(eye, at, up) = cameraEyeAtUpFromPose(m_cameraMatrix, m_rvecs[i], m_tvecs[i]);

		const Camera camera(eye,
			                at,
			                up,
			                fovy,
			                aspectRatio,
			                0.001f,
			                10.0f);

		// Add the camera to the list of cameras
		m_cameras.push_back(camera);
	}
}

void MainWindow::setupPhysicalCameras()
{
	const auto imageWidth = m_imagesRaw.front().cols;
	const auto imageHeight = m_imagesRaw.front().rows;
	
	// Shift image according to the center of the optical axis in the image
	const float imageCenterWidth = getImageCenterX(m_cameraMatrix);
	const float imageCenterHeight = getImageCenterY(m_cameraMatrix);
	const float imageShiftWidth = imageWidth / 2 - imageCenterWidth;
	const float imageShiftHeight = imageHeight / 2 - imageCenterHeight;

	// Pose of cameras in world coordinates
	for (unsigned int i = 0; i < m_cameras.size(); i++)
	{
		// Shift the image according to the optical center
		const auto shiftedImage = translateImage(m_images[i], imageShiftWidth, imageShiftHeight);

		// Add the physical camera to the view
		auto physicalCamera = std::make_unique<PhysicalCamera>(
			m_cameras[i],
			0.05f,
			convertToQtImage(shiftedImage)
			);
		m_ui.viewerWidget->addObject(std::move(physicalCamera));
	}
}

void MainWindow::reconstructPoints()
{
	// Load key points from a file
	m_keypoints.load(m_directory + "keypoints.txt");

	// Homography matrices of each views
	std::vector<cv::Mat1f> homographies;
	for (unsigned int i = 0; i < m_images.size(); i++)
	{
		homographies.push_back(computeProjectionMatrix(m_cameraMatrix, m_rvecs[i], m_tvecs[i]));
	}

	for (unsigned int i = 0; i < m_keypoints.size(); i++)
	{
		std::vector<cv::Mat1f> keypointHomographies;
		std::vector<cv::Vec2f> keypointPoints;

		const auto& keypoint = m_keypoints.getPointsInImages(i);
		
		for (const auto& point : keypoint)
		{
			keypointHomographies.push_back(homographies[point.first]);
			keypointPoints.push_back(point.second);
		}
		
		const auto point = reconstructPointFromViews(keypointHomographies, keypointPoints);

		m_reconstructedPoints.push_back(convertToQt(point));
	}

	// Display points in 2D views
	auto pointObjects = std::make_unique<PointObject>(QMatrix4x4(), m_reconstructedPoints);
	m_ui.viewerWidget->addObject(std::move(pointObjects));
}

void MainWindow::reconstructMesh(int imageId, bool removeLastMesh)
{
	// Get all the keypoints from the nearest view to the current camera view
	const auto imageKeypoints = m_keypoints.getPointInImage(imageId);

	// Triangulate the keypoints in 2D
	std::vector<QVector3D> vertices;
	std::vector<QVector2D> keypointsInImage;
	std::vector<QVector2D> uv;

	// Reserve memory
	vertices.reserve(m_reconstructedPoints.size());
	keypointsInImage.reserve(m_reconstructedPoints.size());
	uv.reserve(m_reconstructedPoints.size());

	// Transform the keypoint of the current image so that we can triangulate and display them
	for (const auto& imageKeypoint : imageKeypoints)
	{
		const auto& coordinatesInImage = imageKeypoint.first;
		const auto& keypointIndex = imageKeypoint.second;

		vertices.push_back(m_reconstructedPoints[keypointIndex]);

		keypointsInImage.emplace_back(coordinatesInImage[0], coordinatesInImage[1]);

		uv.emplace_back(
			coordinatesInImage[0] / m_images[imageId].cols,
			// The origin for UV mapping is bottom left, instead of top left (in Qt)
			(m_images[imageId].rows - coordinatesInImage[1]) / m_images[imageId].rows
		);
	}

	std::vector<std::tuple<int, int, int>> faces = triangulate(keypointsInImage);

	// We remove the last mesh if necessary
	if (removeLastMesh)
	{
		m_ui.viewerWidget->removeLastObject();
	}
	
	// Use the 3D positions of points to get a 3D mesh
	auto mesh = std::make_unique<MeshObject>(QMatrix4x4(), vertices, uv, faces, convertToQtImage(m_images[imageId]));
	m_ui.viewerWidget->addObject(std::move(mesh));
}
