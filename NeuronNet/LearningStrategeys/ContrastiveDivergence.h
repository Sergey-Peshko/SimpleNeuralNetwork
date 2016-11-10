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

		vector<vector<float>> nablaWeights(network->HiddenLayers().size());
		vector<float> nablaThresholdsInput(network->InputLayer());
		vector<float> nablaThresholdsOutput(network->HiddenLayers().size());

		do{
			//выполняем k итераций

			//меняем синоптические связи

			//вычисляем среднеквадратичную ошибку

			epochNumber++;
		} while (epochNumber < _config.getMaxEpoches() &&
			currentError > _config.getMinError() &&
			abs(currentError - lastError) > _config.getMinErrorChange());
	}
}