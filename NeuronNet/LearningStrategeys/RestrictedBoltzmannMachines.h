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
		ILayer* _typeLayer;

		void shuffle(vector<int>& arr);
		vector<float> calculateInvertedOut(ILayer* layer, vector<float>& invertedOut);
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
		vector<vector<float>> nablaWeights(network->Layers()[0]->Neurons().size());
		vector<float> nablaThresholdsOut(network->Layers()[0]->Neurons().size());
		vector<float> nablaThresholdsIn(network->InputThresholds().size());
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
				//îáíóëåíèå îøèáîê ãğóïïû
				for (int j = 0; j < network->Layers()[0]->Neurons().size(); j++)
				{
					for (int k = 0; k < network->Layers()[0]->Neurons()[j]->Weights().size(); k++)
					{
						nablaWeights[j][k] = 0;
					}
					nablaThresholdsOut[j] = 0;
				}
				for (size_t j = 0; j < network->Layers()[0]->getInputDimension(); j++) {
					nablaThresholdsIn[j] = 0;
				}

				//process one batch
				for (int inBatchIndex = currentIndex; inBatchIndex < (currentIndex + _config.getBatchSize()) && inBatchIndex < data.size(); inBatchIndex++)
				{
					vector<float> currRealOutput;
					vector<float> nextRealOutput;

					vector<float> currRealInput;
					vector<float> nextRealInput;

					currRealInput = data[trainingIndices[inBatchIndex]].Input();//x(0)
					currRealOutput = network->Layers()[0]->calculate(currRealInput);//y(0)
					nextRealInput = calculateInvertedOut(network->Layers()[0], currRealOutput);//x(1)
					nextRealOutput = network->Layers()[0]->calculate(nextRealInput);//y(1)

					for (size_t i = 0; i < network->Layers()[0]->Neurons().size(); i++) {
						nablaThresholdsOut[i] +=
							(nextRealOutput[i] - currRealOutput[i]);// * F'(Sj(1))

						for (size_t j = 0; j < network->Layers()[0]->Neurons()[i]->Weights().size(); j++) {

						}
					}

					for (size_t i = 1; i < _paramCD; i++) {
						currRealInput = nextRealOutput;//x(t)
						currRealOutput = network->Layers()[0]->calculate(currRealInput);//y(t)
						nextRealInput = calculateInvertedOut(network->Layers()[0], currRealOutput);//x(t+1)
						nextRealOutput = network->Layers()[0]->calculate(nextRealInput);//y(t+1)
					}

				}
				//update weights and threshold
				for (size_t neuronIndex = 0; neuronIndex < network->Layers()[0]->Neurons().size(); neuronIndex++) {

					for (int weightIndex = 0;
						weightIndex < network->Layers()[0]->Neurons()[neuronIndex]->Weights().size();
						weightIndex++)
					{

					}
				}
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