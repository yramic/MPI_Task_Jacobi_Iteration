#pragma once

#ifndef THE_FILE_NAME_H
#define THE_FILE_NAME_H

#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include <string>
#include <assert.h>
#include <stdio.h>

#endif

///////////////////////////////////////////////////////////////////////////////////////
//FUNCTIONS AND NORMS
///////////////////////////////////////////////////////////////////////////////////////

double ParticularSolution(double x, double y)
{
    return std::sin(2 * M_PI * x) * std::sinh(2 * M_PI * y);
}

double NormL2(const std::vector<double> &v)
{
    double norm = 0;
    for (const auto &value : v)
    {
        norm += value * value;
    }
    return sqrt(norm);
}

double NormInf(const std::vector<double> &v)
{
    double max = std::numeric_limits<double>::lowest();
    for (const auto &value : v)
    {
        max = std::fabs(value) > max ? std::fabs(value) : max;
    }
    return max;
}