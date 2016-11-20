#pragma once
#include "..\stdafx.h"
#include "ILearningStrategey.h";
#include "..\NeuralNetworks\MultilayerNeuralNetwork\IMultilayerNeuralNetwork.h"
#include"Configs\RestrictedBoltzmannMachinesConfig.h"
#include"..\ActivationFunctions\Linear.h"
#include "..\NeuralNetworks\RecurentNeuralNetwork\OLRNN.h"
#include "ContrastiveDivergence.h"
namespace neuralNet {
	class RestrictedBoltzmannMachines : public ILearningStrategy<IMultilayerNeuralNetwork> {
	private:
		RestrictedBoltzmannMachinesConfig _config;
		std::ofstream _logger;
	public:
		// Óíàñëåäîâàíî ÷åğåç ILearningStrategy
		virtual void train(IMultilayerNeuralNetwork * network, vector<DataItem<float>>& data) override;
	};
	void RestrictedBoltzmannMachines::train(IMultilayerNeuralNetwork * network, vector<DataItem<float>>& data) {
		//ÍÀÑÒĞÎÉÊÀ ÏĞÅÄÎÁÓ×ÍÈß
		_logger << ("RBM Start pre-learning...") << std::endl;
		vector<DataItem<float>> tmpData(data);
		for (int i = 0; i < network->HiddenLayers().size(); i++) {
			int dtStart = clock();
			OLRNN tmp(network->HiddenLayers()[i], new ContrastiveDivergence());
			
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