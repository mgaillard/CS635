#pragma once

#include <vector>

#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>

#include "Renderable.h"

class PointObject final : public Renderable
{
public:
	PointObject(const QMatrix4x4& worldMatrix, std::vector<QVector3D> points);

	~PointObject() = default;

	void initialize(QOpenGLContext * context) override;

	void cleanup() override;

	void paint(QOpenGLContext * context, const Camera & camera) override;

private:

	void initializeVbo();

	std::unique_ptr<QOpenGLShaderProgram> m_program;

	QOpenGLVertexArrayObject m_vao;
	QOpenGLBuffer m_vbo;

	std::vector<QVector3D> m_points;
};

