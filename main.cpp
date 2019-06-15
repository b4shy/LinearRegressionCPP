#include <iostream>
#include "PolynomialRegression.h"


int main() {

    std::vector<float> X_train = {1,2,3,4};
    std::vector<float> y_train = {2,5,10,17};
    int degree = 3;
    PolynomialRegression polyreg= PolynomialRegression(X_train, y_train);
    polyreg.fit(degree, 100000 ,0.002);

    polyreg.get_weights();
    float pred = polyreg.predict(10);
    std::cout << pred << std::endl;

    return 0;
}