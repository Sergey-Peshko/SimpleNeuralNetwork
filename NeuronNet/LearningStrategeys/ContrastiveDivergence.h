#pragma once
#include "ILearningStrategey.h"
#include "..\NeuralNetworks\RecurentNeuralNetwork\IRecurentNeuralNetwork.h"
namespace neuralNet {
	class ContrastiveDivergence : public ILearningStrategy<IRecurentNeuralNetwork> {
		ContrastiveDivergenceAlgorithmConfig _config;
		std::ofstream _logger;

	public:
		ContrastiveDivergence();
		// Унаследовано через ILearningStrategy
		virtual void train(IRecurentNeuralNetwork * network, vector<DataItem<float>>& data) override;
	};
	void ContrastiveDivergence::train(IRecurentNeuralNetwork * network, vector<DataItem<float>>& data)
	{
	}
}