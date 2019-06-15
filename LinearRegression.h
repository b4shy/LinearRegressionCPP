//
// Created by marci on 14.06.19.
//

#ifndef MLCPP_LINEARREGRESSION_H
#define MLCPP_LINEARREGRESSION_H

#include <vector>
#include <math.h>
#include <iostream>

class LinearRegression {
public:
    LinearRegression(std::vector<float> data, std::vector<float> labels):
                        X_train(data), y_train(labels){
        std::cout << "Linear Regression with initialized " << std::endl;
    }

    std::vector<float> fit(int degree);
    float predict(float datapoint);
    std::vector<float> get_weights(){
        return weights;
    }



private:
    std::vector<float> X_train;
    std::vector<float> y_train;
    std::vector<float> weights;

    float calculate_error(float datapoint, float label);
    float calculate_loss();
    float update_parameters(int degree);
    void initialize_weights(int degree);


};


#endif //MLCPP_LINEARREGRESSION_H
