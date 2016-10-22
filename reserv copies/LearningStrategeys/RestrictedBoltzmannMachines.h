#pragma once
#include "stdafx.h"
#include "ILearningStrategey.h";
#include "IMultilayerNeuralNetwork.h"
namespace neuralNet {
	class RestrictedBoltzmannMachines : public ILearningStrategy<IMultilayerNeuralNetwork> {
	public:
		// ������������ ����� ILearningStrategy
		virtual void train(IMultilayerNeuralNetwork * network, vector<DataItem<float>*>* data) override;
	};
}