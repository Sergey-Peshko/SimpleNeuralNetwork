#pragma once
#include"IMultilayerNeuralNetwork.h"
#include "..\..\Layers\Layer.h"
#include "..\..\Layers\ILayer.h"
#include"..\..\LearningStrategeys\ILearningStrategey.h"
namespace neuralNet {
	class MLP : public IMultilayerNeuralNetwork {
	private:
		vector<ILayer*> _hiddenLayers;

		ILayer* _outputLayer;
		
		ILearningStrategy<IMultilayerNeuralNetwork>* _learningStrategy;
		//ILearningStrategy<IMultilayerNeuralNetwork>* _pre_learningStrategy;

	public:
		MLP(size_t inputDimension, 
			vector<size_t> hiddenLayersSizes,
			size_t outputDimension,
			IActivationFunction* hidden, 
			IActivationFunction* out,
			ILearningStrategy<IMultilayerNeuralNetwork>* _learningStrategy);
		~MLP();
		// Унаследовано через IMultilayerNeuralNetwork
		virtual vector<float> calculateOutput(vector<float> inputVector) override;
		virtual void save(std::string way) override;
		virtual void train(vector<DataItem<float>>& data) override;
		virtual vector<ILayer*>& HiddenLayers() override;

		virtual ILayer* OutputLayer() override;
	};
	MLP::MLP(size_t inputDimension, 
		vector<size_t> hiddenLayersSizes, 
		size_t outputDimension, 
		IActivationFunction* hidden, 
		IActivationFunction* out,
		ILearningStrategy<IMultilayerNeuralNetwork>* learningStrategy) :
		_hiddenLayers(hiddenLayersSizes.size())
	{

		if (!hiddenLayersSizes.empty()) {
			_hiddenLayers[0] = new Layer(inputDimension, hiddenLayersSizes[0], hidden);
			for (size_t i = 1; i < hiddenLayersSizes.size(); i++) {
				_hiddenLayers[i] = new Layer(hiddenLayersSizes[i - 1], hiddenLayersSizes[i], hidden);
			}

			_outputLayer = new Layer(hiddenLayersSizes[hiddenLayersSizes.size() - 1],
				outputDimension,
				out);
		}
		else {
			_outputLayer = new Layer(inputDimension,
				outputDimension,
				out);
		}
		_learningStrategy = learningStrategy;
	}
	MLP::~MLP() {
		for (size_t i = 0; i < _hiddenLayers.size(); i++) {
			delete _hiddenLayers[i];
		}
		delete _outputLayer;
	}
	vector<float> MLP::calculateOutput(vector<float> inputVector)
	{
		for (size_t i = 0; i < _hiddenLayers.size(); i++) {
			inputVector = _hiddenLayers[i]->calculate(inputVector);
		}
		return _outputLayer->calculate(inputVector);
	}

	void MLP::save(std::string way)
	{
	}

	void MLP::train(vector<DataItem<float>>& data)
	{
		_learningStrategy->train(this, data);
	}

	vector<ILayer*>& MLP::HiddenLayers()
	{
		return _hiddenLayers;
	}

	ILayer* MLP::OutputLayer() {
		return _outputLayer;
	}
}