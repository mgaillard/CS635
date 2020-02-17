#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_MainWindow.h"

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	explicit MainWindow(QWidget *parent = Q_NULLPTR);

public slots:
	void setupScene();

	void selectCameraClicked(int camera);
	
private:
	void setupUi();

	void setupPattern();
	
	void setupCameras();
	
	Ui::MainWindowClass m_ui;

	std::vector<Camera> m_cameras;
};
