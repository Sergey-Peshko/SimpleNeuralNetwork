#pragma once
#include "..\stdafx.h"
#include "..\LearningStrategeys\ILearningStrategey.h"
#include"..\NeuralNetworks\MultilayerNeuralNetwork\IMultilayerNeuralNetwork.h"
#include"Configs\BackpropagationLearningAlgorithmConfig.h"
#include "..\Data\MNISTReader.h"


namespace neuralNet {
	class BackpropagationLearningAlgorithm : public ILearningStrategy<IMultilayerNeuralNetwork> {
	private:
		BackpropagationLearningAlgorithmConfig _config;
		std::ofstream _logger;

		void shuffle(vector<int>& arr);
	public:
		BackpropagationLearningAlgorithm();
		BackpropagationLearningAlgorithm(BackpropagationLearningAlgorithmConfig config);
		~BackpropagationLearningAlgorithm();
		// ”наследовано через ILearningStrategy
		virtual void train(IMultilayerNeuralNetwork* network, vector<DataItem<float>>& data) override;
	};
	BackpropagationLearningAlgorithm::BackpropagationLearningAlgorithm() {
		std::ostringstream ss;
		time_t seconds = time(NULL); // получить текущую дату, выраженную в секундах
		ss << "logsBPA(data" << (int)seconds << ").log";
		_logger = std::ofstream(ss.str());
	}
	BackpropagationLearningAlgorithm::BackpropagationLearningAlgorithm(BackpropagationLearningAlgorithmConfig config) :
		BackpropagationLearningAlgorithm() {

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
		//network->HiddenLayers()[network->HiddenLayers().size() - 1]->Neurons()[network->HiddenLayers().size() - 1]->Weights()[network->HiddenLayers().size() - 1] = -0.5;
		//network->HiddenLayers()[network->HiddenLayers().size() - 1]->Neurons()[network->HiddenLayers().size() - 1]->Weights()[1] = 0.5;
		//network->HiddenLayers()[network->HiddenLayers().size() - 1]->Neurons()[1]->Weights()[network->HiddenLayers().size() - 1] = 0.5;
		//network->HiddenLayers()[network->HiddenLayers().size() - 1]->Neurons()[1]->Weights()[1] = -0.5;

		//network->OutputLayer()->Neurons()[network->HiddenLayers().size() - 1]->Weights()[network->HiddenLayers().size() - 1] = 1;
		//network->OutputLayer()->Neurons()[network->HiddenLayers().size() - 1]->Weights()[1] = 1;

		//MNISTReader rd;
		//vector<DataItem<float>> test = rd.LoadData("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 10'000);

		if (_config.getBatchSize() < 1 || _config.getBatchSize() > data.size())
		{
			_config.setBatchSize(data.size());
		}
	
		float currRegError;
		float currentTestSetError;
		float currentError = FLT_MAX;
		float lastError = 0;
		int epochNumber = 0;
		_logger << ("BPA Start learning...") << std::endl;

		//#region initialize accumulated error for batch, for weights and biases

		vector<vector<vector<float>>> nablaWeights(network->HiddenLayers().size()) ;
		vector<vector<float>> nablaThresholds(network->HiddenLayers().size());

		for (int i = 0; i < network->HiddenLayers().size(); i++)
		{
			nablaWeights[i].resize(network->HiddenLayers()[i]->Neurons().size());
			nablaThresholds[i].resize(network->HiddenLayers()[i]->Neurons().size());
			for (int j = 0; j < network->HiddenLayers()[i]->Neurons().size(); j++)
			{
				nablaWeights[i][j].resize(network->HiddenLayers()[i]->Neurons()[j]->Weights().size());
			}
		}

		vector<vector<float>> nablaWeightsOfOut(network->OutputLayer()->Neurons().size());
		vector<float> nablaThresholdsOfOut(network->OutputLayer()->Neurons().size());

		for (int j = 0; j < nablaWeightsOfOut.size(); j++)
		{
			nablaWeightsOfOut[j].resize(network->OutputLayer()->Neurons()[j]->Weights().size());
		}



		//#endregion

		vector<int> trainingIndices(data.size());
		for (int i = 0; i < data.size(); i++)
		{
			trainingIndices[i] = i;
		}

		do
		{
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
				for (int i = 0; i < nablaWeights.size(); i++)
				{
					for (int j = 0; j < nablaWeights[i].size(); j++)
					{
						for (int k = 0; k < nablaWeights[i][j].size(); k++)
						{
							nablaWeights[i][j][k] = 0;
						}
						nablaThresholds[i][j] = 0;
					}
				}

				for (int i = 0; i < nablaWeightsOfOut.size(); i++) {
					for (int j = 0; j < nablaWeightsOfOut[i].size(); j++) {
						nablaWeightsOfOut[i][j] = 0;
					}
					nablaThresholdsOfOut[i] = 0;
				}

					//process one batch
					for (int inBatchIndex = currentIndex; inBatchIndex < (currentIndex + _config.getBatchSize()) && inBatchIndex < data.size(); inBatchIndex++)
					{

						//forward pass
						vector<float> realOutput = network->calculateOutput(data[trainingIndices[inBatchIndex]].Input());


						//backward pass, error propagation
						//last layer
						//.......................................ќЅ–јЅќ“ ј ѕќ—Ћ≈ƒЌ≈√ќ —Ћќя
						for (int j = 0; j < network->OutputLayer()->Neurons().size(); j++)
						{
							network->OutputLayer()->Neurons()[j]->LastError() =
								_config.getErrorFunction()->calculatePartialDerivaitve(
									data[trainingIndices[inBatchIndex]].Output(),
									realOutput, j) *
								network->OutputLayer()->Neurons()[j]->ActivationFunction()->
								calculateFirstDerivative(network->OutputLayer()->Neurons()[j]->getLastSum());

							nablaThresholdsOfOut[j] += 
								network->OutputLayer()->Neurons()[j]->LastError();

							for (int i = 0; i < network->OutputLayer()->Neurons()[j]->Weights().size(); i++)
							{
								nablaWeightsOfOut[j][i] +=
									network->OutputLayer()->Neurons()[j]->LastError() *
									(network->HiddenLayers().size() > 0 ?
									network->HiddenLayers()[network->HiddenLayers().size() - 1]->Neurons()[i]->getLastState() :
										data[trainingIndices[inBatchIndex]].Input()[i]);

							}
						}

						//hidden layers
						//.......................................ќЅ–јЅќ“ ј — –џ“џ’ —Ћќ≈¬
						for (int hiddenLayerIndex = network->HiddenLayers().size() - 1; hiddenLayerIndex >= 0; hiddenLayerIndex--)
						{
							for (int j = 0; j < network->HiddenLayers()[hiddenLayerIndex]->Neurons().size(); j++)
							{
								network->HiddenLayers()[hiddenLayerIndex]->Neurons()[j]->LastError() = 0;
								if (hiddenLayerIndex == network->HiddenLayers().size() - 1) {
									for (int k = 0; k < network->OutputLayer()->Neurons().size(); k++)
									{
										network->HiddenLayers()[hiddenLayerIndex]->Neurons()[j]->LastError() +=
											network->OutputLayer()->Neurons()[k]->Weights()[j] *
											network->OutputLayer()->Neurons()[k]->LastError();
									}
								}
								else {
									for (int k = 0; k < network->HiddenLayers()[hiddenLayerIndex + 1]->Neurons().size(); k++)
									{
										network->HiddenLayers()[hiddenLayerIndex]->Neurons()[j]->LastError() +=
											network->HiddenLayers()[hiddenLayerIndex + 1]->Neurons()[k]->Weights()[j] *
											network->HiddenLayers()[hiddenLayerIndex + 1]->Neurons()[k]->LastError();
									}
								}
								network->HiddenLayers()[hiddenLayerIndex]->Neurons()[j]->LastError() *=
									network->HiddenLayers()[hiddenLayerIndex]->Neurons()[j]->ActivationFunction()->
									calculateFirstDerivative(
										network->HiddenLayers()[hiddenLayerIndex]->Neurons()[j]->getLastSum()
									);

								nablaThresholds[hiddenLayerIndex][j] +=
									network->HiddenLayers()[hiddenLayerIndex]->Neurons()[j]->LastError();

								for (int i = 0; i < network->HiddenLayers()[hiddenLayerIndex]->Neurons()[j]->Weights().size(); i++)
								{
									nablaWeights[hiddenLayerIndex][j][i] +=
										network->HiddenLayers()[hiddenLayerIndex]->Neurons()[j]->LastError() *
										(hiddenLayerIndex > 0 ?
											network->HiddenLayers()[hiddenLayerIndex - 1]->Neurons()[i]->getLastState() :
											data[trainingIndices[inBatchIndex]].Input()[i]);

								}
							}
						}

					}

				//update weights and bias
				for (int layerIndex = 0; layerIndex < network->HiddenLayers().size(); layerIndex++)
				{
					//_logger << "layer: " << layerIndex << std::endl;
					for (int neuronIndex = 0; 
						neuronIndex < network->HiddenLayers()[layerIndex]->Neurons().size(); 
						neuronIndex++)
					{
						
						network->HiddenLayers()[layerIndex]->Neurons()[neuronIndex]->Threshold() =
							network->HiddenLayers()[layerIndex]->Neurons()[neuronIndex]->Threshold()*
							(1 - _config.getRegularizationFactor()) +	//примен€ем регул€ризацию
							_config.getLearningRate() * nablaThresholds[layerIndex][neuronIndex];
						//_logger << "T: "<< network->HiddenLayers()[layerIndex]->Neurons()[neuronIndex]->Threshold() << "\t\t";
						for (int weightIndex = 0; 
							weightIndex < network->HiddenLayers()[layerIndex]->Neurons()[neuronIndex]->Weights().size();
							weightIndex++)
						{
							network->HiddenLayers()[layerIndex]->Neurons()[neuronIndex]->Weights()[weightIndex] =
								network->HiddenLayers()[layerIndex]->Neurons()[neuronIndex]->Weights()[weightIndex]*
								(1 - _config.getRegularizationFactor()) - //примен€ем регул€ризацию
								_config.getLearningRate() * nablaWeights[layerIndex][neuronIndex][weightIndex];
							//_logger << network->HiddenLayers()[layerIndex]->Neurons()[neuronIndex]->Weights()[weightIndex] << "\t\t";
						}
						//_logger << std::endl;
					}
				}


				//update synaptic conections
				for (int neuronIndex = 0;
					neuronIndex < network->OutputLayer()->Neurons().size();
					neuronIndex++)
				{

					network->OutputLayer()->Neurons()[neuronIndex]->Threshold() =
						network->OutputLayer()->Neurons()[neuronIndex]->Threshold()*
						(1 - _config.getRegularizationFactor()) +	//примен€ем регул€ризацию
						_config.getLearningRate() * nablaThresholdsOfOut[neuronIndex];
					//_logger << "T: "<< network->OutputLayer()->Neurons()[neuronIndex]->Threshold() << "\t\t";
					for (int weightIndex = 0;
						weightIndex < network->OutputLayer()->Neurons()[neuronIndex]->Weights().size();
						weightIndex++)
					{
						network->OutputLayer()->Neurons()[neuronIndex]->Weights()[weightIndex] =
							network->OutputLayer()->Neurons()[neuronIndex]->Weights()[weightIndex] *
							(1 - _config.getRegularizationFactor()) - //примен€ем регул€ризацию
							_config.getLearningRate() * nablaWeightsOfOut[neuronIndex][weightIndex];
						//_logger << network->OutputLayer()->Neurons()[neuronIndex]->Weights()[weightIndex] << "\t\t";
					}
					//_logger << std::endl;
				}


				currentIndex += _config.getBatchSize();

			} while (currentIndex < data.size());

			//recalculating error on all data
			//real error
			currentError = 0;
			for (int i = 0; i < data.size(); i++)
			{
				vector<float> realOutput = network->calculateOutput(data[i].Input());
				currentError += _config.getErrorFunction()->calculate(data[i].Output(), realOutput);
			}
			//regularization term
			
				currRegError = 0;
				for (int layerIndex = 0; layerIndex < network->HiddenLayers().size(); layerIndex++)
				{
					for (int neuronIndex = 0; neuronIndex < network->HiddenLayers()[layerIndex]->Neurons().size(); neuronIndex++)
					{
						for (int weightIndex = 0; weightIndex < network->HiddenLayers()[layerIndex]->Neurons()[neuronIndex]->Weights().size(); weightIndex++)
						{
							currRegError += network->HiddenLayers()[layerIndex]->Neurons()[neuronIndex]->Weights()[weightIndex] *
								network->HiddenLayers()[layerIndex]->Neurons()[neuronIndex]->Weights()[weightIndex];
						}
					}
				}

				for (int neuronIndex = 0; neuronIndex < network->OutputLayer()->Neurons().size(); neuronIndex++)
				{
					for (int weightIndex = 0; weightIndex < network->OutputLayer()->Neurons()[neuronIndex]->Weights().size(); weightIndex++)
					{
						currRegError += network->OutputLayer()->Neurons()[neuronIndex]->Weights()[weightIndex] *
							network->OutputLayer()->Neurons()[neuronIndex]->Weights()[weightIndex];
					}
				}

				currRegError = currRegError / 2;
				
				
			
			epochNumber++;

			_logger << "Eposh #" << epochNumber << std::endl
				<< " finished; current error is " << currentError
				<< " current regularization error is " << currRegError
				<< " Summary error is " << currentError + currRegError
				<< "; it takes: " << (clock() - dtStart) << std::endl;
			/*
			std::cout << "Eposh #" << epochNumber << std::endl
				<< " finished; current error is " << currentError
				<< " current regularization error is " << currRegError
				<< " Summary error is " << currentError + currRegError
				<< "; it takes: " << (clock() - dtStart) << std::endl;
				*/

			currentError += currRegError;



		
			
			int count = 0;
			for (int i = 0; i < _config.getTestSet().size(); i++) {
				vector<float> tmp = network->calculateOutput(_config.getTestSet()[i].Input());
				if (_config.getOutputInterpretatorLogic()->compare(tmp, _config.getTestSet()[i].Output()))
					count++;
			}
			currentTestSetError = 1 - (float)count / (float)_config.getTestSet().size();
			_logger << "TestSet error: " << currentTestSetError << endl;
			//cout << "Test error: " << (float)count / (float)test.size() << endl;
			

		} while (epochNumber < _config.getMaxEpoches() &&
			currentError > _config.getMinError() &&
			abs(currentError - lastError) > _config.getMinErrorChange() &&
			_config.getTestSetError() < currentTestSetError);
	}
}
