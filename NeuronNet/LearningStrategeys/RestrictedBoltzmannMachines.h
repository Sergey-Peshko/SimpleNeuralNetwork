#pragma once
#include "..\stdafx.h"
#include "ILearningStrategey.h";
#include "..\NeuralNetworks\MultilayerNeuralNetwork\IMultilayerNeuralNetwork.h"
#include"LearningAlgorithmConfig.h"
namespace neuralNet {
	class RestrictedBoltzmannMachines : public ILearningStrategy<IMultilayerNeuralNetwork> {
	private:
		LearningAlgorithmConfig _config;
		std::ofstream _logger;

		void shuffle(vector<int>& arr);
	public:
		// Унаследовано через ILearningStrategy
		virtual void train(IMultilayerNeuralNetwork * network, vector<DataItem<float>>& data) override;
	};
	void RestrictedBoltzmannMachines::train(IMultilayerNeuralNetwork * network, vector<DataItem<float>>& data) {
		//
		int epochNumber = 0;
		do {

			for (size_t i = 1; network->Layers().size(); i++) {

			}

			epochNumber++;
		} while (_config.getMaxEpochesOfPreTrein > epochNumber);
	}
}