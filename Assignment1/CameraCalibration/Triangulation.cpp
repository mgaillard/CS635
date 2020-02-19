#include "Triangulation.h"

#include <igl/delaunay_triangulation.h>
#include <igl/predicates/predicates.h>

int orient2dPredicates(const double* pa, const double* pb, const double* pc)
{
    const Eigen::Vector2d a(pa[0], pa[1]);
    const Eigen::Vector2d b(pb[0], pb[1]);
    const Eigen::Vector2d c(pc[0], pc[1]);

    const auto result = igl::predicates::orient2d<Eigen::Vector2d>(a, b, c);

    if (result == igl::predicates::Orientation::POSITIVE) {
        return 1;
    } else if (result == igl::predicates::Orientation::NEGATIVE) {
        return -1;
    } else {
        return 0;
    }
}

int inCirclePredicates(const double* pa, const double* pb, const double* pc, const double* pd)
{
    const Eigen::Vector2d a(pa[0], pa[1]);
    const Eigen::Vector2d b(pb[0], pb[1]);
    const Eigen::Vector2d c(pc[0], pc[1]);
    const Eigen::Vector2d d(pd[0], pd[1]);

    const auto result = igl::predicates::incircle(a, b, c, d);

    if (result == igl::predicates::Orientation::INSIDE) {
        return 1;
    } else if (result == igl::predicates::Orientation::OUTSIDE) {
        return -1;
    } else {
        return 0;
    }
}

std::vector<std::tuple<int, int, int>> triangulate(const std::vector<QVector2D>& points)
{
    Eigen::MatrixXd vertices(points.size(), 2);
    Eigen::MatrixXi faces;

    for (unsigned int i = 0; i < points.size(); i++)
    {
        vertices(i, 0) = points[i].x();
        vertices(i, 1) = points[i].y();
    }

    igl::delaunay_triangulation(vertices,
                                orient2dPredicates,
								inCirclePredicates,
                                faces);

    std::vector<std::tuple<int, int, int>> outputFaces;

    for (unsigned int i = 0; i < faces.rows(); i++)
    {
        outputFaces.emplace_back(
            faces(i, 0),
            faces(i, 1),
            faces(i, 2)
        );
    }
	
    return outputFaces;
}
