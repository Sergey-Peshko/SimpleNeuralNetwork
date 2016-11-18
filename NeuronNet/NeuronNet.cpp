// NeuronNet.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"

#include"NeuralNetworks\MultilayerNeuralNetwork\MLP.h"
#include"LearningStrategeys\BackpropagationLearningAlgorithm.h";
#include"ActivationFunctions\Relu.h"
#include"ActivationFunctions\Sigmoid.h"
#include"Data\DataItem.h"
#include "NeuralNetworks\RecurentNeuralNetwork\OLRNN.h"
#include"LearningStrategeys\ContrastiveDivergence.h";
using namespace neuralNet;

void print(vector<float> v) {
	for (int i = 0; i < v.size(); i++) {
		std::cout << v[i] << " ";
	}
	std::cout << std::endl;
}

int main()
{
	
	//MLP mlp(2, { 2 }, 1, new Relu(), new Sigmoid(), new BackpropagationLearningAlgorithm());
	vector<DataItem<float>> data;
	data.push_back(DataItem<float>({ 0,0 }, { 0 }));
	data.push_back(DataItem<float>({ 0,1 }, { 1 }));
	data.push_back(DataItem<float>({ 1,0 }, { 1 }));
	data.push_back(DataItem<float>({ 1,1 }, { 0 }));
	//data.push_back(DataItem<float>({ -1,-1 }, { 0 }));
	//data.push_back(DataItem<float>({ -1,-2 }, { 1 }));
	//data.push_back(DataItem<float>({ 2,-1 }, { 1 }));
	//data.push_back(DataItem<float>({ 2,2 }, { 0 }));
	/*
	mlp.train(data);
	print(mlp.calculateOutput({ 0,0 }));
	print(mlp.calculateOutput({ 0,1 }));
	print(mlp.calculateOutput({ 1,0 }));
	print(mlp.calculateOutput({ 1,1 }));
	*/
	OLRNN rnn(2,1,new Relu(), new ContrastiveDivergence());
	rnn.train(data);
    return 0;
}

