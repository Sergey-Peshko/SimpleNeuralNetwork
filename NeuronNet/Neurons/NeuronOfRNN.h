#pragma once
#include "..\stdafx.h"
#include "..\ActivationFunctions\IActivationFunction.h"
namespace neuralNet {
	class NeuronOfRNN {
	private:
		vector<float*> weights;
		float* threshold;
		float lastState;
		float lastSum;
		IActivationFunction* activationFunction;
		float lastError;

		float calculateSum(const vector<float>& inputVector) ;
	public:
		NeuronOfRNN(vector<float*> weights,
			float* threshold,
			IActivationFunction* activationFunction);
		NeuronOfRNN(const NeuronOfRNN& obj);
		~NeuronOfRNN();
		 vector<float*>& Weights() ;
		 float * Threshold() ;
		 float activate(const vector<float>& inputVector) ;
		 float getLastState() ;
		 float getLastSum() ;
		 IActivationFunction * ActivationFunction() ;
		 float & LastError() ;
	};
	NeuronOfRNN::NeuronOfRNN(vector<float*> weights,
		float* threshold,
		IActivationFunction* activationFunction)
	{
		this->weights = weights;
		this->threshold = threshold;
		this->activationFunction = activationFunction->clone();
	}
	
	NeuronOfRNN::NeuronOfRNN(const NeuronOfRNN& obj) :
		weights(obj.weights),
		threshold(obj.threshold),
		lastState(obj.lastState),
		lastSum(obj.lastSum),
		lastError(obj.lastError)
	{
		activationFunction = obj.activationFunction->clone();
	}
	
	NeuronOfRNN::~NeuronOfRNN() {
		delete activationFunction;
	}

	vector<float*>& NeuronOfRNN::Weights()
	{
		return weights;
	}

	float * NeuronOfRNN::Threshold()
	{
		return threshold;
	}

	float NeuronOfRNN::calculateSum(const vector<float>& inputVector)
	{
		float sum = 0;
		for (size_t i = 0; i < weights.size(); i++) {
			sum += *(weights[i]) * inputVector[i];
		}
		sum -= *threshold;
		lastSum = sum;
		return sum;
	}

	float NeuronOfRNN::activate(const vector<float>& inputVector)
	{
		lastState = activationFunction->calculate(calculateSum(inputVector));
		return lastState;
	}

	float NeuronOfRNN::getLastState()
	{
		return lastState;
	}

	float NeuronOfRNN::getLastSum()
	{
		return lastSum;
	}

	IActivationFunction * NeuronOfRNN::ActivationFunction()
	{
		return activationFunction;
	}

	float & NeuronOfRNN::LastError()
	{
		return lastError;
	}
}