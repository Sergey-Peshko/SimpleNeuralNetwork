#pragma once
#include "..\stdafx.h"
#include "ILearningStrategey.h";
#include "..\NeuralNetworks\MultilayerNeuralNetwork\IMultilayerNeuralNetwork.h"
#include"LearningAlgorithmConfig.h"
#include"..\ActivationFunctions\Linear.h"
namespace neuralNet {
	class RestrictedBoltzmannMachines : public ILearningStrategy<IMultilayerNeuralNetwork> {
	private:
		BackpropagationLearningAlgorithmConfig _config;
		std::ofstream _logger;
		int _paramCD;	//k параметр CD-k правила

		void shuffle(vector<int>& arr);
	public:
		// Унаследовано через ILearningStrategy
		virtual void train(IMultilayerNeuralNetwork * network, vector<DataItem<float>>& data) override;
	};
	void RestrictedBoltzmannMachines::train(IMultilayerNeuralNetwork * network, vector<DataItem<float>>& data) {
		//НАСТРОЙКА ПРЕДОБУЧНИЯ
		if (_config.getBatchSize() < 1 || _config.getBatchSize() > data.size())
		{
			_config.setBatchSize(data.size());
		}
		vector<int> trainingIndices(data.size());
		for (int i = 0; i < data.size(); i++)
		{
			trainingIndices[i] = i;
		}
		//ОБЪЯВЛЕНИЕ ГЛАБАЛЬНЫХ ПЕРЕМЕННЫХ
		float currentError;
		float lastError;
		int epochNumber;
		//
		_logger << ("RBM Start learning...") << std::endl;
		//.................ОБРАБОТКА ПЕРВОГО СКРЫТОГО СОЛЯ
		currentError = FLT_MAX;
		lastError = 0;
		epochNumber = 0;
		vector<vector<float>> nablaWeights(network->HiddenLayers()[0]->Neurons().size());
		vector<float> nablaThresholdsOut(network->HiddenLayers()[0]->Neurons().size());
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
				//обнуление ошибок группы
				for (int j = 0; j < network->HiddenLayers()[0]->Neurons().size(); j++)
				{
					for (int k = 0; k < network->HiddenLayers()[0]->Neurons()[j]->Weights().size(); k++)
					{
						nablaWeights[j][k] = 0;
					}
					nablaThresholdsOut[j] = 0;
				}
				for (size_t j = 0; j < network->HiddenLayers()[0]->getInputDimension(); j++) {
					nablaThresholdsIn[j] = 0;
				}

				//process one batch
				for (int inBatchIndex = currentIndex; inBatchIndex < (currentIndex + _config.getBatchSize()) && inBatchIndex < data.size(); inBatchIndex++)
				{
					vector<float> currOutput;
					vector<float> nextOutput;

					vector<float> currInput;
					vector<float> nextInput;

					vector<float> sumsOfNextInput; //последние суммы неронов 0-ой слой i слой

					//currInput = data[trainingIndices[inBatchIndex]].Input();//x(0)
					//снос поргов, но пока пороги равны 0 и поэтому сноса нет
					currOutput = network->HiddenLayers()[0]->calculate(data[trainingIndices[inBatchIndex]].Input());//y(0)
					calculateInvertedOut(nextInput, sumsOfNextInput, network->HiddenLayers()[0], currOutput, new Linear());//x(1)
					nextOutput = network->HiddenLayers()[0]->calculate(nextInput);//y(1)

					//просчет наблов
					for (size_t j = 0; j < network->HiddenLayers()[0]->Neurons().size(); j++) {
						nablaThresholdsOut[j] +=
							(nextOutput[j] - currOutput[j]) *
							network->HiddenLayers()[0]->Neurons()[j]->ActivationFunction()->calculateFirstDerivative(
								network->HiddenLayers()[0]->Neurons()[j]->getLastSum());// * F'(Sj(1))

						for (size_t i = 0; i < network->HiddenLayers()[0]->Neurons()[j]->Weights().size(); i++) {
							nablaWeights[j][i] +=
								(nextOutput[j] - currOutput[j]) *
								nextInput[i] *
								network->HiddenLayers()[0]->Neurons()[j]->ActivationFunction()->calculateFirstDerivative(
									network->HiddenLayers()[0]->Neurons()[j]->getLastSum())
								+
								(nextInput[i] - data[trainingIndices[inBatchIndex]].Input()[i])*
								currOutput[j] *
								sumsOfNextInput[i];// * F'(Si(1))
						}
					}
					for (size_t i = 0; i < nablaThresholdsIn.size(); i++) {
						nablaThresholdsIn[i] +=
							(nextInput[i] - data[trainingIndices[inBatchIndex]].Input()[i])*
							sumsOfNextInput[i];// * F'(Si(1))
					}

					for (size_t i = 1; i < _paramCD; i++) {
						currInput = nextInput;//x(0)
						//снос поргов, но пока пороги равны 0 и поэтому сноса нет
						currOutput = network->HiddenLayers()[0]->calculate(currInput);//y(0)
						calculateInvertedOut(nextInput, sumsOfNextInput, network->HiddenLayers()[0], currOutput, new Linear());//x(1)
						nextOutput = network->HiddenLayers()[0]->calculate(nextInput);//y(1)

						//просчет наблов
						for (size_t j = 0; j < network->HiddenLayers()[0]->Neurons().size(); j++) {
							nablaThresholdsOut[j] +=
								(nextOutput[j] - currOutput[j]) *
								network->HiddenLayers()[0]->Neurons()[j]->ActivationFunction()->calculateFirstDerivative(
									network->HiddenLayers()[0]->Neurons()[j]->getLastSum());// * F'(Sj(1))

							for (size_t i = 0; i < network->HiddenLayers()[0]->Neurons()[j]->Weights().size(); i++) {
								nablaWeights[j][i] +=
									(nextOutput[j] - currOutput[j]) *
									nextInput[i] *
									network->HiddenLayers()[0]->Neurons()[j]->ActivationFunction()->calculateFirstDerivative(
										network->HiddenLayers()[0]->Neurons()[j]->getLastSum())
									+
									(nextInput[i] - data[trainingIndices[inBatchIndex]].Input()[i])*
									currOutput[j] *
									sumsOfNextInput[i];// * F'(Si(1))
							}
						}
						for (size_t i = 0; i < nablaThresholdsIn.size(); i++) {
							nablaThresholdsIn[i] +=
								(nextInput[i] - data[trainingIndices[inBatchIndex]].Input()[i])*
								sumsOfNextInput[i];// * F'(Si(1))
						}
					}

				}
				//update weights and threshold
				//update weights and threshold of j layer 
				for (size_t neuronIndex = 0; neuronIndex < network->HiddenLayers()[0]->Neurons().size(); neuronIndex++) {
					network->HiddenLayers()[0]->Neurons()[neuronIndex]->Threshold() -=
						_config.getLearningRate() * nablaThresholdsOut[neuronIndex];
					for (int weightIndex = 0;
						weightIndex < network->HiddenLayers()[0]->Neurons()[neuronIndex]->Weights().size();
						weightIndex++)
					{
						network->HiddenLayers()[0]->Neurons()[neuronIndex]->Weights()[weightIndex] -=
							_config.getLearningRate() * nablaWeights[neuronIndex][weightIndex];
					}
				}
				//threshold of i layer 
				for (size_t i = 0; i < network->InputThresholds().size(); i++) {
					network->InputThresholds()[i] -= nablaThresholdsIn[i];
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

		//.................ОБРАБОТКА ОСТАЛЬНЫХ СКРЫТЫХ СЛОЕВ
		for (size_t i = 1; network->HiddenLayers().size(); i++) {
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