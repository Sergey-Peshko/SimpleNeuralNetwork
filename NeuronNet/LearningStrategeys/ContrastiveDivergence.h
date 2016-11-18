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
		// ”наследовано через ILearningStrategy
		virtual void train(IRecurentNeuralNetwork * network, vector<DataItem<float>>& data) override;
	};
	ContrastiveDivergence::ContrastiveDivergence() {
		std::ostringstream ss;
		time_t seconds = time(NULL); // получить текущую дату, выраженную в секундах
		ss << "logsCD(data" << (int)seconds << ").log" << std::endl;
		std::string lol = ss.str();
		std::string way(lol.begin(), lol.end() - 1);
		_logger = std::ofstream(way);
	}
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
			lastError = currentError;
			int dtStart = clock();
			//process data set
			currentError = 0;
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
				//выполн€ем k итераций
				for (int i = 0; i < _config.getK(); i++) {
					finishInput = network->calculateInput(prevOutput);
					finishOutput = network->calculateOutput(finishInput);
					//прибавл€ем к наблам
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
									network->getInvertedLayer()->Neurons()[weightIndex].getLastSum());
						}
						nablaThresholdsOutput[neuronIndex] += (finishOutput[neuronIndex] - prevOutput[neuronIndex]) *
							network->OutputLayer()->Neurons()[neuronIndex]->ActivationFunction()->calculateFirstDerivative(
								network->OutputLayer()->Neurons()[neuronIndex]->getLastSum());
					}
					for (int weightIndex = 0; weightIndex < nablaThresholdsInput.size(); weightIndex++) {
						nablaThresholdsInput[weightIndex] +=
							(finishInput[weightIndex] - prevInput[weightIndex]) *
							network->getInvertedLayer()->Neurons()[weightIndex].ActivationFunction()->calculateFirstDerivative(
								network->getInvertedLayer()->Neurons()[weightIndex].getLastSum());
					}
					//вычисл€ем среднеквадратичную ошибку
					currentError += _config.ErrorFunction()->calculate(finishInput, prevInput);
					currentError += _config.ErrorFunction()->calculate(finishOutput, prevOutput);

					prevOutput = finishOutput;
					prevInput = finishInput;
				}
				
				//мен€ем синоптические св€зи
				for (int i = 0; i < nablaWeights.size(); i++)
				{
					for (int j = 0; j < nablaWeights[i].size(); j++)
					{
						network->OutputLayer()->Neurons()[i]->Weights()[j] -= _config.getLearningRate() * nablaWeights[i][j];
					}
					network->OutputLayer()->Neurons()[i]->Threshold() -= _config.getLearningRate() * nablaThresholdsOutput[i];
				}
				for (int i = 0; i < nablaThresholdsInput.size(); i++) {
					network->getInvertedLayer()->Neurons()[i]->Threshold() -= _config.getLearningRate() * nablaThresholdsInput[i];
				}


				currentIndex++;
			} while (currentIndex < data.size());
			
			vector<float> startInput = data[currentIndex].Input();
			vector<float> startOutput = network->calculateOutput(startInput);

			vector<float> finishInput;
			vector<float> finishOutput;

			vector<float> prevOutput = startOutput;
			vector<float> prevInput = startInput;

			//вычисл€ем среднеквадратичную ошибку
			//выполн€ем k итераций
			for (int i = 0; i < data.size(); i++)
			{
				for (int i = 0; i < _config.getK(); i++) {
					finishInput = network->calculateInput(prevOutput);
					finishOutput = network->calculateOutput(finishInput);

					currentError += _config.ErrorFunction()->calculate(finishInput, prevInput);
					currentError += _config.ErrorFunction()->calculate(finishOutput, prevOutput);

					prevOutput = finishOutput;
					prevInput = finishInput;
				}
			}

			_logger << "Eposh #" << epochNumber << " finished;" << std::endl
				<< "current error is " << currentError
				<< "; it takes: " << (clock() - dtStart) << std::endl;

			epochNumber++;
		} while (epochNumber < _config.getMaxEpoches() &&
			currentError > _config.getMinError() &&
			abs(currentError - lastError) > _config.getMinErrorChange());
	}
}