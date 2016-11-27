#pragma once
#include"IMultilayerNeuralNetwork.h"
#include "..\..\Layers\Layer.h"
#include "..\..\Layers\ILayer.h"
#include"..\..\LearningStrategeys\ILearningStrategey.h"
#include"..\..\LearningStrategeys\BackpropagationLearningAlgorithm.h"
#include"..\..\LearningStrategeys\RestrictedBoltzmannMachines.h"
#include"..\..\ActivationFunctions\Relu.h"
#include"..\..\ActivationFunctions\Sigmoid.h"
namespace neuralNet {
	class MLP : public IMultilayerNeuralNetwork {
	private:
		vector<ILayer*> _hiddenLayers;

		ILayer* _outputLayer;
		
		ILearningStrategy<IMultilayerNeuralNetwork>* _learningStrategy;
		ILearningStrategy<IMultilayerNeuralNetwork>* _preLearningStrategy;

	public:
		MLP(){}
		MLP(size_t inputDimension, 
			vector<size_t> hiddenLayersSizes,
			size_t outputDimension,
			IActivationFunction* hidden, 
			IActivationFunction* out,
			ILearningStrategy<IMultilayerNeuralNetwork>* _learningStrategy);
		MLP(size_t inputDimension,
			vector<size_t> hiddenLayersSizes,
			size_t outputDimension,
			IActivationFunction* hidden,
			IActivationFunction* out,
			ILearningStrategy<IMultilayerNeuralNetwork>* _learningStrategy,
			ILearningStrategy<IMultilayerNeuralNetwork>* _preLearningStrategy);
		MLP::MLP(vector<ILayer*> hidden,
			ILayer* out);
		~MLP();
		
		void open(std::string way);
		void setLearningStrategy(ILearningStrategy<IMultilayerNeuralNetwork>* learningStrategy);
		void setPreLearningStrategy(ILearningStrategy<IMultilayerNeuralNetwork>* preLearningStrategy);
		

		// Унаследовано через IMultilayerNeuralNetwork
		virtual vector<float> calculateOutput(vector<float> inputVector) override;
		virtual void save(std::string way) override;
		virtual void train(vector<DataItem<float>>& data) override;
		virtual void preTrain(vector<DataItem<float>>& data);
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

	MLP::MLP(vector<ILayer*> hidden,
		ILayer* out) :
		_hiddenLayers(hidden),
		_outputLayer(out)
	{
	}

	MLP::MLP(size_t inputDimension,
		vector<size_t> hiddenLayersSizes,
		size_t outputDimension,
		IActivationFunction* hidden,
		IActivationFunction* out,
		ILearningStrategy<IMultilayerNeuralNetwork>* learningStrategy,
		ILearningStrategy<IMultilayerNeuralNetwork>* preLearningStrategy) :
		MLP::MLP(inputDimension,
			hiddenLayersSizes,
			outputDimension,
			hidden,
			out,
			learningStrategy) {
		_preLearningStrategy = preLearningStrategy;
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
		ofstream os(way);
		os << _hiddenLayers.size() << endl;
		for (int i = 0; i < _hiddenLayers.size(); i++) {
			os << _hiddenLayers[i]->toString() << endl;
		}

		os << _outputLayer->toString() << endl;
	}
	
	void MLP::open(std::string way)
	{
		ifstream is(way);
		int hiddenLayersSize;
		is >> hiddenLayersSize;
		_hiddenLayers.resize(hiddenLayersSize);

		for (int i = 0; i < _hiddenLayers.size(); i++) {
			int neurons_size;
			int input_demension;
			is >> neurons_size;
			is >> input_demension;
			vector<vector<float>> w(neurons_size, vector<float>(input_demension));
			vector<float> t(neurons_size);
			for (int i = 0; i < neurons_size; i++) {
				for (int j = 0; j < input_demension; j++) {
					is >> w[i][j];
				}
				is >> t[i];
			}

			_hiddenLayers[i] = new Layer(w, t, new Relu());
		}

		int neurons_size;
		int input_demension;
		is >> neurons_size;
		is >> input_demension;
		vector<vector<float>> w(neurons_size, vector<float>(input_demension));
		vector<float> t(neurons_size);
		for (int i = 0; i < neurons_size; i++) {
			for (int j = 0; j < input_demension; j++) {
				is >> w[i][j];
			}
			is >> t[i];
		}
		_outputLayer = new Layer(w, t, new Sigmoid());
	}
	void MLP::setLearningStrategy(ILearningStrategy<IMultilayerNeuralNetwork>* learningStrategy) {
		_learningStrategy = learningStrategy;
	}
	void MLP::setPreLearningStrategy(ILearningStrategy<IMultilayerNeuralNetwork>* preLearningStrategy) {
		_preLearningStrategy = preLearningStrategy;
	}

	void MLP::train(vector<DataItem<float>>& data)
	{
		_learningStrategy->train(this, data);
	}
	void MLP::preTrain(vector<DataItem<float>>& data) {
		_preLearningStrategy->train(this, data);
	}
	vector<ILayer*>& MLP::HiddenLayers()
	{
		return _hiddenLayers;
	}

	ILayer* MLP::OutputLayer() {
		return _outputLayer;
	}
}