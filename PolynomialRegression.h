//
// Created by marci on 14.06.19.
//

#ifndef MLCPP_POLYNOMIALREGRESSION_H
#define MLCPP_POLYNOMIALREGRESSION_H

#include <vector>
#include <math.h>
#include <iostream>

class PolynomialRegression {
public:

    PolynomialRegression(std::vector<float> data, std::vector<float> labels):
                        X_train(data), y_train(labels){
        std::cout << "Linear Regression" << std::endl;
    }

    void fit(int degree, int numsteps, float lr);
    float predict(float datapoint);
    std::vector<float> get_weights(){
        return weights;
    }



private:
    std::vector<float> X_train;
    std::vector<float> y_train;
    std::vector<float> weights;

    //float calculate_error(float datapoint, float label);
    float calculate_loss();
    float update_parameters(int degree, float lr);
    void initialize_weights(int degree);
    void print_training_status();
};


#endif //MLCPP_POLYNOMIALREGRESSION_H
