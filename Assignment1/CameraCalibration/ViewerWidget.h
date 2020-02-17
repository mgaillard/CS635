#pragma once

#include <QOpenGLWidget>
#include <QOpenGLFunctions_4_3_Core>
#include <QOpenGLDebugLogger>

#include "Camera.h"
#include "Renderable.h"

class ViewerWidget : public QOpenGLWidget, protected QOpenGLFunctions_4_3_Core
{
	Q_OBJECT

public:
	explicit ViewerWidget(QWidget* parent = Q_NULLPTR);
	virtual ~ViewerWidget();

	ViewerWidget(const ViewerWidget& widget) = delete;
	ViewerWidget& operator=(ViewerWidget other) = delete;
	ViewerWidget(ViewerWidget&&) = delete;
	ViewerWidget& operator=(ViewerWidget&&) = delete;

	void addObject(std::unique_ptr<Renderable> object);

	void setCamera(const OrbitCamera& camera);

public slots:
	void cleanup();
	void printInfo();

protected:
	void initializeGL() override;
	void resizeGL(int w, int h) override;
	void paintGL() override;
	void mousePressEvent(QMouseEvent* event) override;
	void mouseReleaseEvent(QMouseEvent* event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void wheelEvent(QWheelEvent* event) override;

private:
	QOpenGLDebugLogger* m_logger;

	OrbitCamera m_camera;

	std::vector<std::unique_ptr<Renderable>> m_objects;
};
