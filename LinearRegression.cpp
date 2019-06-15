//
// Created by marci on 14.06.19.
//

#include "LinearRegression.h"
#include <math.h>
#include <cassert>
#include <algorithm>

std::vector<float>LinearRegression::fit(int degree){

    assert(X_train.size() == y_train.size());
    std::cout << "Fitting LinReg with a Polynomial of " << degree << std::endl;
    initialize_weights(degree);

    for(int i = 0; i < 2000; ++i){
        float loss = calculate_loss();
        std::cout << loss << std::endl;
        update_parameters(degree);
    }
    for(auto i : weights){
        std::cout << i << " ";
    }
}


float LinearRegression::predict(float datapoint) {
    float result = 0;

    for(int i = 0; i != weights.size(); ++i){
        result+=(weights[i] * pow(datapoint, i));
    }

    return result;
}


float LinearRegression::calculate_loss() {
    float loss = 0;
    for(int i = 0; i != X_train.size(); ++i){
        float prediction = predict(X_train[i]);
        std::cout << "Pred: " << prediction << " ";

        float actual_loss = calculate_error(prediction, y_train[i]);
        loss+=actual_loss;
    }
    return loss;
}

float LinearRegression::calculate_error(float datapoint, float label){
    return pow(datapoint - label, 2);
}

float LinearRegression::update_parameters(int degree) {
    std::vector<float> dM(degree, 0);


    for(int i = 1; i != degree; ++i){
        for(int j = 0; j != X_train.size(); ++j){
            dM[i] += ((-2*i*pow(X_train[j],std::max(0,i)))*(y_train[j] - predict(X_train[j]))) / X_train.size();

        }
    }

    for (int i = 0; i != weights.size(); ++i) {
        weights[i] -= 0.005 * dM[i];
    }

}


void LinearRegression::initialize_weights(int degree) {
    weights.resize(degree,0);
}


