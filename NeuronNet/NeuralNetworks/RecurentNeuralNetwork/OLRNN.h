#pragma once
#include "..\..\stdafx.h"
#include "IRecurentNeuralNetwork.h"
#include"..\..\LearningStrategeys\ILearningStrategey.h"
#include"..\..\Layers\InvertedLayer.h"
#include "..\..\Layers\InputLayer.h"
#include "..\..\Layers\Layer.h"

namespace neuralNet {
	class OLRNN : public IRecurentNeuralNetwork {
		ILayer* _inputLayer;
		ILayer* _outputLayer;

		InvertedLayer _invertedLayer;

		ILearningStrategy<IRecurentNeuralNetwork>* _learningStrategy;
	public:
		OLRNN(size_t inputDimension,
			size_t outputDimension,
			IActivationFunction* out,
			ILearningStrategy<IRecurentNeuralNetwork>* _learningStrategy);
		// Унаследовано через IRecurentNeuralNetwork
		virtual ILayer * InputLayer() override;
		virtual ILayer * OutputLayer() override;
		virtual vector<float> calculateOutput(vector<float> inputVector) override;
		virtual vector<float> calculateInput(vector<float> outputVector) override;
		virtual void save(std::string way) override;
		virtual void train(vector<DataItem<float>>& data) override;
	};

	OLRNN::OLRNN(size_t inputDimension,
		size_t outputDimension,
		IActivationFunction* out,
		ILearningStrategy<IRecurentNeuralNetwork>* _learningStrategy) {
		_inputLayer = new InputLayer(inputDimension);
		_outputLayer = new Layer(inputDimension, outputDimension, out);

		_invertedLayer = InvertedLayer(_inputLayer, _outputLayer);
	}

	void neuralNet::OLRNN::save(std::string way)
	{
	}

	void neuralNet::OLRNN::train(vector<DataItem<float>>& data)
	{
		_learningStrategy->train(this, data);
	}

	ILayer * neuralNet::OLRNN::InputLayer()
	{
		return _inputLayer;
	}

	ILayer * neuralNet::OLRNN::OutputLayer()
	{
		return _outputLayer;
	}

	vector<float> neuralNet::OLRNN::calculateOutput(vector<float> inputVector)
	{
		inputVector = _inputLayer->calculate(inputVector);

		return _outputLayer->calculate(inputVector);
	}

	vector<float> neuralNet::OLRNN::calculateInput(vector<float> outputVector)
	{
		return _invertedLayer.calculate(outputVector);
	}
}