#pragma once
#include "stdafx.h"
#include "ILearningStrategey.h"
#include"IMultilayerNeuralNetwork.h"
namespace neuralNet {
	class BackpropagationLearningAlgorithm : ILearningStrategy<IMultilayerNeuralNetwork> {
		// ������������ ����� ILearningStrategy
		virtual void train(IMultilayerNeuralNetwork* network, vector<DataItem<float>*>* data) override;
	};
}