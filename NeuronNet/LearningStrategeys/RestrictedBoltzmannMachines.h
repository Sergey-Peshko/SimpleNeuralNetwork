#pragma once
#include "..\stdafx.h"
#include "ILearningStrategey.h";
#include "..\NeuralNetworks\MultilayerNeuralNetwork\IMultilayerNeuralNetwork.h"
#include"Configs\ContrastiveDivergenceAlgorithmConfig.h"
#include"..\ActivationFunctions\Linear.h"
#include "..\NeuralNetworks\RecurentNeuralNetwork\OLRNN.h"
#include "ContrastiveDivergence.h"
namespace neuralNet {
	class RestrictedBoltzmannMachines : public ILearningStrategy<IMultilayerNeuralNetwork> {
	private:
		ContrastiveDivergenceAlgorithmConfig _config;
		std::ofstream _logger;
	public:
		RestrictedBoltzmannMachines();
		RestrictedBoltzmannMachines(ContrastiveDivergenceAlgorithmConfig _config);
		// ”наследовано через ILearningStrategy
		virtual void train(IMultilayerNeuralNetwork * network, vector<DataItem<float>>& data) override;
	};
	RestrictedBoltzmannMachines::RestrictedBoltzmannMachines() {
		std::ostringstream ss;
		time_t seconds = time(NULL); // получить текущую дату, выраженную в секундах
		ss << "logsRBM(data" << (int)seconds << ").log";
		_logger = std::ofstream(ss.str());
	}
	RestrictedBoltzmannMachines::RestrictedBoltzmannMachines(ContrastiveDivergenceAlgorithmConfig config) :
		RestrictedBoltzmannMachines()
	{
		_config = config;
	}
	void RestrictedBoltzmannMachines::train(IMultilayerNeuralNetwork * network, vector<DataItem<float>>& data) {
		
		_logger << ("RBM Start pre-learning...") << std::endl;
		vector<DataItem<float>> tmpData(data);
		for (int i = 0; i < network->HiddenLayers().size(); i++) {
			int dtStart = clock();
			std::ostringstream ss;
			ss << i;
			OLRNN tmp(network->HiddenLayers()[i], new ContrastiveDivergence(_config, ss.str()));
			
			for (int j = 0; j < i; j++) {
				for (int k = 0; k < data.size(); k++) {
					tmpData[k].Input() = network->HiddenLayers()[j]->calculate(tmpData[k].Input());
				}
			}

			tmp.train(tmpData);
			_logger << "hidden layer #" << i << 
				" finish pre-train; it takes " << clock() - dtStart << std::endl;
		}
	}
}