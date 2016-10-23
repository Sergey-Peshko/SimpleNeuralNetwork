#pragma once
#include "..\stdafx.h"
#include "..\LearningStrategeys\ILearningStrategey.h"
#include"..\NeuralNetworks\MultilayerNeuralNetwork\IMultilayerNeuralNetwork.h"
#include"LearningAlgorithmConfig.h"

namespace neuralNet {
	class BackpropagationLearningAlgorithm : public ILearningStrategy<IMultilayerNeuralNetwork> {
	private:
		LearningAlgorithmConfig _config;
		std::ofstream _logger;

		void shuffle(vector<int>& arr);
	public:
		BackpropagationLearningAlgorithm();
		BackpropagationLearningAlgorithm(LearningAlgorithmConfig config);
		~BackpropagationLearningAlgorithm();
		// ”наследовано через ILearningStrategy
		virtual void train(IMultilayerNeuralNetwork* network, vector<DataItem<float>>& data) override;
	};
	BackpropagationLearningAlgorithm::BackpropagationLearningAlgorithm() {
		std::ostringstream ss;
		time_t seconds = time(NULL); // получить текущую дату, выраженную в секундах
		ss << "logs(data" << (int)seconds << ").log" << std::endl;
		std::string lol = ss.str();
		std::string way(lol.begin(), lol.end() - 1);
		_logger = std::ofstream(way);
	}
	BackpropagationLearningAlgorithm::BackpropagationLearningAlgorithm(LearningAlgorithmConfig config) {
		std::ostringstream ss;
		time_t seconds = time(NULL); // получить текущую дату, выраженную в секундах
		ss << "logs(data" << (int)seconds << ").log" << std::endl;
		std::string lol = ss.str();
		std::string way(lol.begin(), lol.end() - 1);
		_logger = std::ofstream(way);

		_config = config;
	}
	BackpropagationLearningAlgorithm::~BackpropagationLearningAlgorithm() {
		_logger.close();
	}
	void BackpropagationLearningAlgorithm::shuffle(vector<int>& arr)
	{
		std::mt19937 _generator(clock());
		std::uniform_real_distribution<double> _urd(0, 1);
		std::uniform_int_distribution<int> _uid(0, arr.size() - 1);

		for (int i = 0; i < arr.size() - 1; i++)
		{
			if (_urd(_generator) >= 0.3)
			{
				int newIndex = _uid(_generator);
				int tmp = arr[i];
				arr[i] = arr[newIndex];
				arr[newIndex] = tmp;
			}
		}
	}
	void neuralNet::BackpropagationLearningAlgorithm::train(IMultilayerNeuralNetwork* network, vector<DataItem<float>>& data)
	{
	//	network->Layers()[0]->Neurons()[0]->Weights()[0] = -0.5;
	//	network->Layers()[0]->Neurons()[0]->Weights()[1] = 0.5;
	//	network->Layers()[0]->Neurons()[1]->Weights()[0] = 0.5;
	//	network->Layers()[0]->Neurons()[1]->Weights()[1] = -0.5;

	//	network->Layers()[1]->Neurons()[0]->Weights()[0] = 1;
	//	network->Layers()[1]->Neurons()[0]->Weights()[1] = 1;

		if (_config.getBatchSize() < 1 || _config.getBatchSize() > data.size())
		{
			_config.setBatchSize(data.size());
		}
		float currentError = FLT_MAX;
		float lastError = 0;
		int epochNumber = 0;
		_logger << ("BPA Start learning...") << std::endl;

		//#region initialize accumulated error for batch, for weights and biases

		vector<vector<vector<float>>> nablaWeights(network->Layers().size()) ;
		vector<vector<float>> nablaThresholds(network->Layers().size());

		for (int i = 0; i < network->Layers().size(); i++)
		{
			nablaWeights[i].resize(network->Layers()[i]->Neurons().size());
			nablaThresholds[i].resize(network->Layers()[i]->Neurons().size());
			for (int j = 0; j < network->Layers()[i]->Neurons().size(); j++)
			{
				nablaWeights[i][j].resize(network->Layers()[i]->Neurons()[j]->Weights().size());
			}
		}

		//#endregion

		do
		{
			lastError = currentError;
			int dtStart = clock();

			//preparation for epoche
			vector<int> trainingIndices(data.size());
			for (int i = 0; i < data.size(); i++)
			{
				trainingIndices[i] = i;
			}
			if (_config.getBatchSize() > 0)
			{
				//shuffle(trainingIndices);
			}

			//process data set
			int currentIndex = 0;
			do
			{
				//обнуление ошибок группы
				for (int i = 0; i < network->Layers().size(); i++)
				{
					for (int j = 0; j < network->Layers()[i]->Neurons().size(); j++)
					{
						for (int k = 0; k < network->Layers()[i]->Neurons()[j]->Weights().size(); k++)
						{
							nablaWeights[i][j][k] = 0;
						}
						nablaThresholds[i][j] = 0;
					}
				}

					//process one batch
					for (int inBatchIndex = currentIndex; inBatchIndex < (currentIndex + _config.getBatchSize()) && inBatchIndex < data.size(); inBatchIndex++)
					{

						//forward pass
						vector<float> realOutput = network->calculateOutput(data[trainingIndices[inBatchIndex]].Input());


						//backward pass, error propagation
						//last layer
						//.......................................ќЅ–јЅќ“ ј ѕќ—Ћ≈ƒЌ≈√ќ —Ћќя
						for (int j = 0; j < network->Layers()[network->Layers().size() - 1]->Neurons().size(); j++)
						{
							network->Layers()[network->Layers().size() - 1]->Neurons()[j]->LastError() =
								_config.ErrorFunction()->calculatePartialDerivaitve(
									data[trainingIndices[inBatchIndex]].Output(),
									realOutput, j) *
								network->Layers()[network->Layers().size() - 1]->Neurons()[j]->ActivationFunction()->
								calculateFirstDerivative(network->Layers()[network->Layers().size() - 1]->Neurons()[j]->getLastSum());

							nablaThresholds[network->Layers().size() - 1][j] += 
								network->Layers()[network->Layers().size() - 1]->Neurons()[j]->LastError();

							for (int i = 0; i < network->Layers()[network->Layers().size() - 1]->Neurons()[j]->Weights().size(); i++)
							{
								nablaWeights[network->Layers().size() - 1][j][i] +=
									network->Layers()[network->Layers().size() - 1]->Neurons()[j]->LastError() *
									network->Layers()[network->Layers().size() - 1 - 1]->Neurons()[i]->getLastState();

							}
						}


						//hidden layers
						//.......................................ќЅ–јЅќ“ ј — –џ“џ’ —Ћќ≈¬
						for (int hiddenLayerIndex = network->Layers().size() - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--)
						{
							for (int j = 0; j < network->Layers()[hiddenLayerIndex]->Neurons().size(); j++)
							{
								network->Layers()[hiddenLayerIndex]->Neurons()[j]->LastError() = 0;
								for (int k = 0; k < network->Layers()[hiddenLayerIndex + 1]->Neurons().size(); k++)
								{
									network->Layers()[hiddenLayerIndex]->Neurons()[j]->LastError() +=
										network->Layers()[hiddenLayerIndex + 1]->Neurons()[k]->Weights()[j] *
										network->Layers()[hiddenLayerIndex + 1]->Neurons()[k]->LastError();
								}
								network->Layers()[hiddenLayerIndex]->Neurons()[j]->LastError() *=
									network->Layers()[hiddenLayerIndex]->Neurons()[j]->ActivationFunction()->
									calculateFirstDerivative(
										network->Layers()[hiddenLayerIndex]->Neurons()[j]->getLastSum()
									);

								nablaThresholds[hiddenLayerIndex][j] +=
									network->Layers()[hiddenLayerIndex]->Neurons()[j]->LastError();

								for (int i = 0; i < network->Layers()[hiddenLayerIndex]->Neurons()[j]->Weights().size(); i++)
								{
									nablaWeights[hiddenLayerIndex][j][i] +=
										network->Layers()[hiddenLayerIndex]->Neurons()[j]->LastError() *
										(hiddenLayerIndex > 0 ? 
											network->Layers()[hiddenLayerIndex - 1]->Neurons()[i]->getLastState() : 
											data[trainingIndices[inBatchIndex]].Input()[i]);
										

								}
							}
						}
					}

				//update weights and bias
				for (int layerIndex = 0; layerIndex < network->Layers().size(); layerIndex++)
				{
					//_logger << "layer: " << layerIndex << std::endl;
					for (int neuronIndex = 0; 
						neuronIndex < network->Layers()[layerIndex]->Neurons().size(); 
						neuronIndex++)
					{
						
						network->Layers()[layerIndex]->Neurons()[neuronIndex]->Threshold() =
							network->Layers()[layerIndex]->Neurons()[neuronIndex]->Threshold()*
							(1 - _config.getRegularizationFactor()) +	//примен€ем регул€ризацию
							_config.getLearningRate() * nablaThresholds[layerIndex][neuronIndex];
						//_logger << "T: "<< network->Layers()[layerIndex]->Neurons()[neuronIndex]->Threshold() << "\t\t";
						for (int weightIndex = 0; 
							weightIndex < network->Layers()[layerIndex]->Neurons()[neuronIndex]->Weights().size();
							weightIndex++)
						{
							network->Layers()[layerIndex]->Neurons()[neuronIndex]->Weights()[weightIndex] =
								network->Layers()[layerIndex]->Neurons()[neuronIndex]->Weights()[weightIndex]*
								(1 - _config.getRegularizationFactor()) - //примен€ем регул€ризацию
								_config.getLearningRate() * nablaWeights[layerIndex][neuronIndex][weightIndex];
							//_logger << network->Layers()[layerIndex]->Neurons()[neuronIndex]->Weights()[weightIndex] << "\t\t";
						}
						//_logger << std::endl;
					}
				}

				currentIndex += _config.getBatchSize();

			} while (currentIndex < data.size());

			//recalculating error on all data
			//real error
			currentError = 0;
			for (int i = 0; i < data.size(); i++)
			{
				vector<float> realOutput = network->calculateOutput(data[i].Input());
				currentError += _config.ErrorFunction()->calculate(data[i].Output(), realOutput);
			}
			//regularization term
			/*
				float reg = 0;
				for (int layerIndex = 0; layerIndex < network->Layers().size(); layerIndex++)
				{
					for (int neuronIndex = 0; neuronIndex < network->Layers()[layerIndex]->Neurons().size(); neuronIndex++)
					{
						for (int weightIndex = 0; weightIndex < network->Layers()[layerIndex]->Neurons()[neuronIndex]->Weights().size(); weightIndex++)
						{
							reg += network->Layers()[layerIndex]->Neurons()[neuronIndex]->Weights()[weightIndex] *
								network->Layers()[layerIndex]->Neurons()[neuronIndex]->Weights()[weightIndex];
						}
					}
				}
				reg = reg / 2;
				currentError += reg;
				*/
			
			epochNumber++;

			_logger << "Eposh #" << epochNumber << std::endl
				<< " finished; current error is " << currentError
				<< "; it takes: " << (clock() - dtStart) << std::endl;

		} while (epochNumber < _config.getMaxEpoches() &&
			currentError > _config.getMinError() &&
			abs(currentError - lastError) > _config.getMinErrorChange());
	}
}
