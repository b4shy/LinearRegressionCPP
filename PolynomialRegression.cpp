//
// Created by marci on 14.06.19.
//

#include "PolynomialRegression.h"
#include <math.h>
#include <cassert>
#include <algorithm>

void PolynomialRegression::fit(int degree, int num_steps, float lr){

    assert(X_train.size() == y_train.size());
    std::cout << "Fitting LinReg with a Polynomial of " << degree << std::endl;
    initialize_weights(degree);

    for(int i = 0; i < num_steps; ++i){
        update_parameters(degree, lr);

        if(i% 10000) {
            print_training_status();
        }

    }
}


float PolynomialRegression::predict(float datapoint) {
    float result = 0;
    float bias = weights[0];

    for(int i = 1; i != weights.size(); ++i){
        result+=(weights[i] * pow(datapoint, i));
    }
    result+=bias;

    return result;
}


float PolynomialRegression::update_parameters(int degree, float lr) {
    std::vector<float> dM(degree, 0);

    for(int i = 0; i != degree; ++i){
        for(int j = 0; j != X_train.size(); ++j){
            if(i==0){
                dM[i] = -2*(y_train[j] - predict(X_train[j])) / X_train.size();
            }
            dM[i] += ((-2*i*pow(X_train[j],std::max(0,i)))*(y_train[j] - predict(X_train[j]))) / X_train.size();
        }
    }

    for (int i = 0; i != weights.size(); ++i) {
        weights[i] -= lr * dM[i];
    }

}

void PolynomialRegression::initialize_weights(int degree) {
    weights.resize(degree,0);
}

void PolynomialRegression::print_weights(){

    for(auto i : weights){
        std::cout << "Weights: " << i << std::endl;
    }

}

void PolynomialRegression::print_training_status(){
        float loss = calculate_loss();
        std::cout << "Actual Loss: " << loss << std::endl;
        print_weights();
}


float PolynomialRegression::calculate_loss() {
    float loss = 0;
    for(int i = 0; i != X_train.size(); ++i){
        float prediction = predict(X_train[i]);
        std::cout << "X = " << X_train[i] << " Prediction = " << prediction << " ";
        float actual_loss = calculate_error(prediction, y_train[i]);
        loss+=actual_loss;
    }
    return loss;
}

float PolynomialRegression::calculate_error(float datapoint, float label){
    return pow(datapoint - label, 2);
}
