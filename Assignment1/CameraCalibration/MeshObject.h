#pragma once

#include <vector>

#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QOpenGLTexture>


#include "Renderable.h"

class MeshObject final : public Renderable
{
public:
	MeshObject(const QMatrix4x4& worldMatrix,
		       std::vector<QVector3D> vertices,
		       std::vector<QVector2D> uv,
		       std::vector<std::tuple<int, int, int>> faces,
			   QImage textureImage);

	~MeshObject() = default;

	void initialize(QOpenGLContext* context) override;

	void cleanup() override;

	void paint(QOpenGLContext* context, const Camera& camera) override;

private:

	void initializeVbo();
	void initializeEbo();
	void initializeTexture();

	std::unique_ptr<QOpenGLShaderProgram> m_program;

	QOpenGLVertexArrayObject m_vao;
	QOpenGLBuffer m_vbo;
	QOpenGLBuffer m_ebo;
	QOpenGLTexture m_texture;
	

	std::vector<QVector3D> m_vertices;
	std::vector<QVector2D> m_uv;
	std::vector<std::tuple<int, int, int>> m_faces;
	QImage m_textureImage;
};
