#pragma once
#include"IMultilayerNeuralNetwork.h"
#include "..\..\Layers\Layer.h"
#include "..\..\Layers\ILayer.h"
#include"..\..\LearningStrategeys\ILearningStrategey.h"
namespace neuralNet {
	class MLP : public IMultilayerNeuralNetwork {
	private:
		vector<ILayer*> _layers;
		ILearningStrategy<IMultilayerNeuralNetwork>* _learningStrategy;
		//ILearningStrategy<IMultilayerNeuralNetwork>* _pre_learningStrategy;

		vector<float> _inputThresholds;
	public:
		MLP(size_t inputDimension, vector<size_t> layersSizes, 
			IActivationFunction* hidden, IActivationFunction* out,
			ILearningStrategy<IMultilayerNeuralNetwork>* _learningStrategy);
		~MLP();
		// Унаследовано через IMultilayerNeuralNetwork
		virtual vector<float> calculateOutput(vector<float> inputVector) override;
		virtual void save(std::string way) override;
		virtual void train(vector<DataItem<float>>& data) override;
		virtual vector<ILayer*>& Layers() override;

		virtual vector<float>& InputThresholds() override;
	};
	MLP::MLP(size_t inputDimension, 
		vector<size_t> layersSizes, 
		IActivationFunction* hidden, 
		IActivationFunction* out,
		ILearningStrategy<IMultilayerNeuralNetwork>* learningStrategy) :
		_layers(layersSizes.size()),
		_inputThresholds(inputDimension, 0)
	{
		_layers[0] = new Layer(inputDimension, layersSizes[0], hidden);
		for (size_t i = 1; i < layersSizes.size() - 1; i++) {
			_layers[i] = new Layer(layersSizes[i - 1], layersSizes[i], hidden);
		}
		_layers[layersSizes.size() - 1] = new Layer(layersSizes[layersSizes.size() - 2],
			layersSizes[layersSizes.size() - 1], 
			out);

		_learningStrategy = learningStrategy;
	}
	MLP::~MLP() {
		for (size_t i = 0; i < _layers.size(); i++) {
			delete _layers[i];
		}
	}
	vector<float> MLP::calculateOutput(vector<float> inputVector)
	{
		vector<float> out;
		for (size_t i = 0; i < inputVector.size(); i++) {
			inputVector[i] -= _inputThresholds[i];
		}
		for (size_t i = 0; i < _layers.size(); i++) {
			out = _layers[i]->calculate(inputVector);
			inputVector = out;
		}
		return out;
	}

	void MLP::save(std::string way)
	{
	}

	void MLP::train(vector<DataItem<float>>& data)
	{
		_learningStrategy->train(this, data);
	}

	vector<ILayer*>& MLP::Layers()
	{
		return _layers;
	}

	vector<float>& MLP::InputThresholds() {
		return _inputThresholds;
	}
}