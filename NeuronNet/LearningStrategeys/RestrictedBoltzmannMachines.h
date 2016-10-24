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
		int _paramCD;	//k ïàğàìåòğ CD-k ïğàâèëà

		void shuffle(vector<int>& arr);
	public:
		// Óíàñëåäîâàíî ÷åğåç ILearningStrategy
		virtual void train(IMultilayerNeuralNetwork * network, vector<DataItem<float>>& data) override;
	};
	void RestrictedBoltzmannMachines::train(IMultilayerNeuralNetwork * network, vector<DataItem<float>>& data) {
		//ÍÀÑÒĞÎÉÊÀ ÏĞÅÄÎÁÓ×ÍÈß
		if (_config.getBatchSize() < 1 || _config.getBatchSize() > data.size())
		{
			_config.setBatchSize(data.size());
		}
		vector<int> trainingIndices(data.size());
		for (int i = 0; i < data.size(); i++)
		{
			trainingIndices[i] = i;
		}
		//ÎÁÚßÂËÅÍÈÅ ÃËÀÁÀËÜÍÛÕ ÏÅĞÅÌÅÍÍÛÕ
		float currentError;
		float lastError;
		int epochNumber;
		//
		_logger << ("RBM Start learning...") << std::endl;
		//.................ÎÁĞÀÁÎÒÊÀ ÏÅĞÂÎÃÎ ÑÊĞÛÒÎÃÎ ÑÎËß
		currentError = FLT_MAX;
		lastError = 0;
		epochNumber = 0;
		do {
			lastError = currentError;
			int dtStart = clock();

			//preparation for epoche		
			if (_config.getBatchSize() > 0)
			{
				shuffle(trainingIndices);
			}

			//process data set
			int currentIndex = 0;
			do
			{
				for (int inBatchIndex = currentIndex; inBatchIndex < (currentIndex + _config.getBatchSize()) && inBatchIndex < data.size(); inBatchIndex++)
				{


				}
				//update weights and threshold

				//
				currentIndex += _config.getBatchSize();
			} while (currentIndex < data.size());
			//recalculating error on all data
			currentError = 0;

			//

			epochNumber++;
			_logger << "Layer #" << 0 << "Eposh #" << epochNumber << std::endl
				<< " finished; current error is " << currentError
				<< "; it takes: " << (clock() - dtStart) << std::endl;
		} while (epochNumber < _config.getMaxEpoches() &&
			currentError > _config.getMinError() &&
			abs(currentError - lastError) > _config.getMinErrorChange());
		//.................ÎÁĞÀÁÎÒÊÀ ÎÑÒÀËÜÍÛÕ ÑÊĞÛÒÛÕ ÑËÎÅÂ
		for (size_t i = 1; network->Layers().size(); i++) {
			currentError = FLT_MAX;
			lastError = 0;
			epochNumber = 0;
			do {

				epochNumber++;
			} while (epochNumber < _config.getMaxEpoches() &&
				currentError > _config.getMinError() &&
				abs(currentError - lastError) > _config.getMinErrorChange());
		}

			
	}
}