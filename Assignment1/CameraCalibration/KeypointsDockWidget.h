#pragma once

#include <QDockWidget>
#include "ui_KeypointsDockWidget.h"

class KeypointsDockWidget : public QDockWidget
{
	Q_OBJECT

public:
	KeypointsDockWidget(QWidget *parent = Q_NULLPTR);
	~KeypointsDockWidget();

public slots:
	void addKeypoint(int image, qreal x, qreal y);

private:
	Ui::KeypointsDockWidget m_ui;
};
