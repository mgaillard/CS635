#pragma once

#include <vector>

#include <QVector2D>

/**
 * \brief Wrapper to triangulate 2D vertices using libigl
 */
std::vector<std::tuple<int, int, int>> triangulate(const std::vector<QVector2D>& points);
