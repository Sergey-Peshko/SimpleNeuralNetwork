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

		do{
			lastError = currentError;
			int dtStart = clock();
			//process data set
			int currentIndex = 0;
			do
			{

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

					prevOutput = finishOutput;
					prevInput = finishInput;
				}
				
				//меняем синоптические связи
				for (int i = 0; i <network->OutputLayer()->Neurons().size(); i++)
				{
					for (int j = 0; j < network->OutputLayer()->Neurons()[i]->Weights().size(); j++)
					{
						network->OutputLayer()->Neurons()[i]->Weights()[j] -=
							_config.getLearningRate() * (
								finishInput[j] * finishOutput[i] - startInput[j] * startOutput[i]
								);
					}
					
					network->OutputLayer()->Neurons()[i]->Threshold() +=
						_config.getLearningRate() * (
							finishOutput[i] - startOutput[i]
							);
							
				}
				for (int i = 0; i < network->getInvertedLayer()->Neurons().size(); i++) {
					
					network->getInvertedLayer()->Neurons()[i]->Threshold() +=
						_config.getLearningRate() * (
							finishInput[i] - startInput[i]
							);
							
				}


				currentIndex++;
			} while (currentIndex < data.size());
			
			//вычисляем среднеквадратичную ошибку
			//выполняем k итераций
			currentError = 0;
			for (int i = 0; i < data.size(); i++)
			{
				vector<float> startInput = data[i].Input();
				vector<float> startOutput = network->calculateOutput(startInput);

				vector<float> finishInput;
				vector<float> finishOutput;

				vector<float> prevOutput = startOutput;
				vector<float> prevInput = startInput;

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