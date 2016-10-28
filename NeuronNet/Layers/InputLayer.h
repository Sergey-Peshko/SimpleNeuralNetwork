#pragma once
#include "ILayer.h"
#include"..\ActivationFunctions\IActivationFunction.h"
#include"..\Neurons\InputNeuron.h"
namespace neuralNet {
	class InputLayer : public ILayer {
	private:
		vector<INeuron*> neurons;
		vector<float> lastOut;
		size_t inputDimension;
	public:
		InputLayer(size_t inputDimension);
		InputLayer(const InputLayer& obj);
		~InputLayer();
		// Унаследовано через ILayer
		virtual const vector<float>& calculate(const vector<float>& inputVector) override;
		virtual const vector<float>& LastOutput() override;
		virtual vector<INeuron*>& Neurons() override;
		virtual size_t getInputDimension() override;
		virtual ILayer* clone() override;
	};
	InputLayer::InputLayer(size_t inputDimension) :
		neurons(inputDimension),
		lastOut(inputDimension),
		inputDimension(inputDimension)
	{
		for (size_t i = 0; i < inputDimension; i++) {
			neurons[i] = new InputNeuron(inputDimension, i);
		}
	}
	InputLayer::InputLayer(const InputLayer& obj) :
		inputDimension(obj.inputDimension),
		lastOut(obj.lastOut),
		neurons(obj.neurons.size())
	{
		for (size_t i = 0; i < neurons.size(); i++) {
			neurons[i] = obj.neurons[i]->clone();
		}
	}
	InputLayer::~InputLayer() {
		for (size_t i = 0; i < neurons.size(); i++) {
			delete neurons[i];
		}
	}
	const vector<float>& InputLayer::calculate(const vector<float>& inputVector)
	{
		for (size_t i = 0; i < neurons.size(); i++) {
			lastOut[i] = neurons[i]->activate(inputVector);
		}
		return lastOut;
	}

	const vector<float>& InputLayer::LastOutput()
	{
		return lastOut;
	}

	vector<INeuron*>& InputLayer::Neurons()
	{
		return neurons;
	}

	size_t InputLayer::getInputDimension()
	{
		return inputDimension;
	}
	ILayer* InputLayer::clone() {
		return new InputLayer(*this);
	}
}