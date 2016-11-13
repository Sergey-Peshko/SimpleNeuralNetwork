#pragma once
#include "ILearningStrategey.h"
#include "..\NeuralNetworks\RecurentNeuralNetwork\IRecurentNeuralNetwork.h"
#include "ContrastiveDivergenceAlgorithmConfig.h"
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

		float currentError = FLT_MAX;
		float lastError = 0;
		int epochNumber = 0;
		_logger << ("CD-k Start learning...") << std::endl;

		vector<float> nablaThresholdsInput(network->OutputLayer()->getInputDimension());
		vector<float> nablaThresholdsOutput(network->OutputLayer()->Neurons().size());
		vector<vector<float>> nablaWeights(network->OutputLayer()->Neurons().size(), 
			vector<float>(network->OutputLayer()->getInputDimension()));
		

		do{

			//process data set
			int currentIndex = 0;
			do
			{
				for (int i = 0; i < nablaWeights.size(); i++)
				{
					for (int j = 0; j < nablaWeights[i].size(); j++)
					{
						nablaWeights[i][j] = 0;
					}
					nablaThresholdsOutput[i] = 0;
				}
				for (int i = 0; i < nablaThresholdsInput.size(); i++) {
					nablaThresholdsInput[i] = 0;
				}

				vector<float> startInput = data[currentIndex].Input();
				vector<float> startOutput = network->calculateOutput(startInput);

				vector<float> finishInput;
				vector<float> finishOutput;

				vector<float> prevOutput = startOutput;
				vector<float> prevInput = startInput;
				//выполняем k итераций
				for (int i = 0; i < _config.getK(); i++) {
					finishInput = network->calculateInput(prevOutput);
					finishOutput = network->calculateOutput(finishInput);
					//прибавляем к наблам
					for (int neuronIndex = 0; neuronIndex < nablaWeights.size(); neuronIndex++) {
						for (int weightIndex = 0; weightIndex < nablaWeights[neuronIndex].size(); weightIndex++) {
							nablaWeights[neuronIndex][weightIndex] +=
								(finishOutput[neuronIndex] - prevOutput[neuronIndex]) *
								(finishInput[weightIndex]) *
								network->OutputLayer()->Neurons()[neuronIndex]->ActivationFunction()->calculateFirstDerivative(
									network->OutputLayer()->Neurons()[neuronIndex]->getLastSum())
								+
								(finishInput[weightIndex] - prevInput[weightIndex]) *
								(prevOutput[neuronIndex]) *
								network->OutputLayer()->Neurons()[weightIndex]->ActivationFunction()->calculateFirstDerivative(
									network->getInvertedLayer().Neurons()[weightIndex].getLastSum());
						}
						nablaThresholdsOutput[neuronIndex] += (finishOutput[neuronIndex] - prevOutput[neuronIndex]) *
							network->OutputLayer()->Neurons()[neuronIndex]->ActivationFunction()->calculateFirstDerivative(
								network->OutputLayer()->Neurons()[neuronIndex]->getLastSum());
					}
					for (int weightIndex = 0; weightIndex < nablaThresholdsInput.size(); weightIndex++) {
						nablaThresholdsInput[weightIndex] +=
							(finishInput[weightIndex] - prevInput[weightIndex]) *
							network->getInvertedLayer().Neurons()[weightIndex].ActivationFunction()->calculateFirstDerivative(
								network->getInvertedLayer().Neurons()[weightIndex].getLastSum());
					}

					prevOutput = finishOutput;
					prevInput = finishInput;
				}
				
				//меняем синоптические связи
				for (int i = 0; i < nablaWeights.size(); i++)
				{
					for (int j = 0; j < nablaWeights[i].size(); j++)
					{
						network->OutputLayer()->Neurons()[i]->Weights()[j] -= _config.getLearningRate() * nablaWeights[i][j];
					}
					network->OutputLayer()->Neurons()[i]->Threshold() -= _config.getLearningRate() * nablaThresholdsOutput[i];
				}
				for (int i = 0; i < nablaThresholdsInput.size(); i++) {
					network->OutputLayer()->Neurons()[i]->Threshold() -= _config.getLearningRate() * nablaThresholdsInput[i];
				}

			} while (currentIndex < data.size());
			//вычисляем среднеквадратичную ошибку

			epochNumber++;
		} while (epochNumber < _config.getMaxEpoches() &&
			currentError > _config.getMinError() &&
			abs(currentError - lastError) > _config.getMinErrorChange());
	}
}