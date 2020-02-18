#pragma once

#include <QtWidgets/QMainWindow>

#include "ui_MainWindow.h"

#include <opencv2/core/core.hpp>

#include "KeypointsDockWidget.h"

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	explicit MainWindow(QWidget *parent = Q_NULLPTR);

public slots:
	void setupScene();

	void selectCameraClicked(int camera);

	void viewDoubleClicked(qreal x, qreal y);
	
private:
	void setupUi();

	void setupPattern();
	
	void setupCameras();

	void setupPhysicalCameras();

	void reconstructPoints();
	
	Ui::MainWindowClass m_ui;
	KeypointsDockWidget* m_keypointsDock;

	const std::string m_directory;
	std::vector<cv::Mat> m_imagesRaw;
	std::vector<cv::Mat> m_images;

	cv::Mat1f m_cameraMatrix;
	std::vector<cv::Mat> m_rvecs;
	std::vector<cv::Mat> m_tvecs;

	std::vector<Camera> m_cameras;
	int m_currentCamera;
};
