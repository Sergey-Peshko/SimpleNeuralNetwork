#pragma once
#include "ILayer.h"
#include"..\ActivationFunctions\IActivationFunction.h"
#include"..\Neurons\NeuronofRNN.h"
namespace neuralNet {
	class InvertedLayer {
	private:
		vector<NeuronOfRNN> neurons;
		vector<float> lastOut;
		size_t inputDimension;
	public:
		InvertedLayer();
		InvertedLayer(ILayer* input, ILayer* output);
		InvertedLayer(const InvertedLayer& obj);
		~InvertedLayer();
		const vector<float>& calculate(const vector<float>& inputVector);
		const vector<float>& LastOutput();
		vector<NeuronOfRNN>& Neurons();
		size_t getInputDimension() ;
	};
	InvertedLayer::InvertedLayer() {

	}
	InvertedLayer::InvertedLayer(ILayer* input, ILayer* output)
	{
		inputDimension = output->Neurons().size();

		neurons.resize(output->getInputDimension());

		vector<float*> wights(inputDimension);

		for (int i = 0; i < neurons.size(); i++) {
			float* threshold = &(input->Neurons()[i]->Threshold);
			vector<float*> wights(inputDimension);

			for (int j = 0; j < inputDimension; j++) {
				wights[j] = &(output->Neurons()[j]->Weights()[i]);
			}
			neurons[i] = NeuronOfRNN(wights, threshold, input->Neurons()[i]->ActivationFunction());
		}
	}
	
	InvertedLayer::InvertedLayer(const InvertedLayer& obj) :
		inputDimension(obj.inputDimension),
		lastOut(obj.lastOut),
		neurons(obj.neurons.size())
	{
		for (size_t i = 0; i < neurons.size(); i++) {
			neurons[i] = NeuronOfRNN(obj.neurons[i]);
		}
	}
	
	InvertedLayer::~InvertedLayer() {
		
	}
	const vector<float>& InvertedLayer::calculate(const vector<float>& inputVector)
	{
		for (size_t i = 0; i < neurons.size(); i++) {
			lastOut[i] = neurons[i].activate(inputVector);
		}
		return lastOut;
	}

	const vector<float>& InvertedLayer::LastOutput()
	{
		return lastOut;
	}

	vector<NeuronOfRNN>& InvertedLayer::Neurons()
	{
		return neurons;
	}

	size_t InvertedLayer::getInputDimension()
	{
		return inputDimension;
	}
}