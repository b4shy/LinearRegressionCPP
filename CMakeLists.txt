cmake_minimum_required(VERSION 3.14)
project(mlcpp)

set(CMAKE_CXX_STANDARD 14)

add_executable(mlcpp main.cpp PolynomialRegression.cpp PolynomialRegression.h)