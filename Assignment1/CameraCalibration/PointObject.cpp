#include "PointObject.h"

#include <QOpenGLFunctions_4_3_Core>

PointObject::PointObject(const QMatrix4x4& worldMatrix, std::vector<QVector3D> points) :
	Renderable(worldMatrix),
	m_vbo(QOpenGLBuffer::VertexBuffer),
	m_points(std::move(points))
{
	
}

void PointObject::initialize(QOpenGLContext* context)
{
	const QString shader_dir = ":/MainWindow/Shaders/";

	// Init Program
	m_program = std::make_unique<QOpenGLShaderProgram>();
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, shader_dir + "point_vs.glsl");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, shader_dir + "point_fs.glsl");
	m_program->link();

	m_program->bind();

	// Initialize vertices and indices
	initializeVbo();

	// Init VAO
	m_vao.create();
	QOpenGLVertexArrayObject::Binder vaoBinder(&m_vao);

	// Configure VBO
	m_vbo.bind();
	const auto posLoc = 0;
	m_program->enableAttributeArray(posLoc);
	m_program->setAttributeArray(posLoc, nullptr, 3, 0);

	m_program->release();
}

void PointObject::cleanup()
{
	if (m_program)
	{
		m_vao.destroy();
		m_vbo.destroy();
		m_program.reset(nullptr);
	}
}

void PointObject::paint(QOpenGLContext* context, const Camera& camera)
{
	auto f = context->versionFunctions<QOpenGLFunctions_4_3_Core>();

	if (m_program)
	{
		// Setup matrices
		const auto normalMatrix = worldMatrix().normalMatrix();
		const auto viewMatrix = camera.viewMatrix();
		const auto projectionMatrix = camera.projectionMatrix();
		const auto pvMatrix = projectionMatrix * viewMatrix;
		const auto pvmMatrix = pvMatrix * worldMatrix();

		m_program->bind();

		// Update matrices
		m_program->setUniformValue("P", projectionMatrix);
		m_program->setUniformValue("V", viewMatrix);
		m_program->setUniformValue("M", worldMatrix());
		m_program->setUniformValue("N", normalMatrix);
		m_program->setUniformValue("PV", pvMatrix);
		m_program->setUniformValue("PVM", pvmMatrix);

		// Bind the VAO containing the patches
		QOpenGLVertexArrayObject::Binder vaoBinder(&m_vao);

		f->glEnable(GL_PROGRAM_POINT_SIZE);
		f->glDrawArrays(GL_POINTS, 0, m_vbo.size());
		f->glDisable(GL_PROGRAM_POINT_SIZE);

		m_program->release();
	}
}

void PointObject::initializeVbo()
{
	// Init VBO
	m_vbo.create();
	m_vbo.bind();

	m_vbo.allocate(m_points.data(), m_points.size() * sizeof(QVector3D));

	m_vbo.release();
}
