#pragma once
#include "stdafx.h"
#include "ILearningStrategey.h";
#include "IMultilayerNeuralNetwork.h"
namespace neuralNet {
	class RestrictedBoltzmannMachines : ILearningStrategy<IMultilayerNeuralNetwork> {
		// Унаследовано через ILearningStrategy
		virtual void train(IMultilayerNeuralNetwork * network, vector<DataItem<float>*>* data) override;
	};
}