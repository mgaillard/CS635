#include "KeypointsDockWidget.h"

KeypointsDockWidget::KeypointsDockWidget(QWidget *parent)
	: QDockWidget(parent)
{
	m_ui.setupUi(this);
}

KeypointsDockWidget::~KeypointsDockWidget()
{
	
}

void KeypointsDockWidget::addKeypoint(int image, qreal x, qreal y)
{
	const auto keypointString = QString::number(image) + " " + QString::number(x) + " " + QString::number(y);
	
	m_ui.textEdit->append(keypointString);
}
