#pragma once
#include "ILayer.h"
#include"..\ActivationFunctions\IActivationFunction.h"
#include"..\Neurons\NeuronofRNN.h"
namespace neuralNet {
	class InvertedLayer : public ILayer {
	private:
		vector<INeuron*> neurons;
		vector<float> lastOut;
		size_t inputDimension;
	public:
		InvertedLayer();
		InvertedLayer(ILayer* output);
		InvertedLayer(const InvertedLayer& obj);
		~InvertedLayer();
		const vector<float>& calculate(const vector<float>& inputVector);
		const vector<float>& LastOutput();
		vector<INeuron*>& Neurons();
		size_t getInputDimension() ;
		ILayer* clone();
	};
	InvertedLayer::InvertedLayer() {

	}
	InvertedLayer::InvertedLayer(ILayer* output)
	{
		inputDimension = output->Neurons().size();

		neurons.resize(output->getInputDimension());
		lastOut.resize(output->getInputDimension());

		for (int i = 0; i < neurons.size(); i++) {
			vector<float*> wights(inputDimension);

			for (int j = 0; j < inputDimension; j++) {
				wights[j] = &(output->Neurons()[j]->Weights()[i]);
			}
			neurons[i] = new NeuronOfRNN(wights, output->Neurons()[0]->ActivationFunction());
		}
	}
	
	InvertedLayer::InvertedLayer(const InvertedLayer& obj) :
		inputDimension(obj.inputDimension),
		lastOut(obj.lastOut),
		neurons(obj.neurons.size())
	{
		for (size_t i = 0; i < neurons.size(); i++) {
			neurons[i] = obj.neurons[i]->clone();
		}
	}
	
	InvertedLayer::~InvertedLayer() {
		
	}
	const vector<float>& InvertedLayer::calculate(const vector<float>& inputVector)
	{
		for (size_t i = 0; i < neurons.size(); i++) {
			lastOut[i] = neurons[i]->activate(inputVector);
		}
		return lastOut;
	}

	const vector<float>& InvertedLayer::LastOutput()
	{
		return lastOut;
	}

	vector<INeuron*>& InvertedLayer::Neurons()
	{
		return neurons;
	}

	size_t InvertedLayer::getInputDimension()
	{
		return inputDimension;
	}
	ILayer* InvertedLayer::clone() {
		return new InvertedLayer(*this);
	}
}