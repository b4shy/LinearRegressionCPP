#include <iostream>
#include "LinearRegression.h"


int main() {
    std::cout << "Hello, World!" << std::endl;

    std::vector<float> X_train = {1,2,3,4};
    std::vector<float> y_train = {1,4,9,16};
    int degree = 3;
    LinearRegression linreg = LinearRegression(X_train, y_train);
    linreg.fit(degree);

    linreg.get_weights();
    float pred = linreg.predict(7.0);
    std::cout << pred << std::endl;




    return 0;
}