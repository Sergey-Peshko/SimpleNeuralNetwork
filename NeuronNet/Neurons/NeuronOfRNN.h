#pragma once
#include "..\stdafx.h"
#include "..\ActivationFunctions\IActivationFunction.h"
#include "INeuron.h"
namespace neuralNet {
	class NeuronOfRNN : public INeuron {
	private:
		vector<float*> weights;
		vector<float> Vweights;
		float threshold;
		float lastState;
		float lastSum;
		IActivationFunction* activationFunction;
		float lastError;

		float calculateSum(const vector<float>& inputVector) ;
	public:
		NeuronOfRNN(vector<float*> weights,
			IActivationFunction* activationFunction);
		NeuronOfRNN(const NeuronOfRNN& obj);
		~NeuronOfRNN();
		 vector<float>& Weights() ;
		 float & Threshold() ;
		 float activate(const vector<float>& inputVector) ;
		 float getLastState() ;
		 float getLastSum() ;
		 IActivationFunction * ActivationFunction() ;
		 float & LastError() ;

		 // Унаследовано через INeuron
		 virtual INeuron * clone() override;
	};

	NeuronOfRNN::NeuronOfRNN(vector<float*> weights,
		IActivationFunction* activationFunction)
	{
		this->weights = weights;
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

	vector<float>& NeuronOfRNN::Weights()
	{
		vector<float> rez(weights.size());
		for (int i = 0; i < rez.size(); i++) {
			rez[i] = *(weights[i]);
		}
		Vweights = rez;
		return Vweights;
	}

	float&  NeuronOfRNN::Threshold()
	{
		return threshold;
	}

	float NeuronOfRNN::calculateSum(const vector<float>& inputVector)
	{
		float sum = 0;
		for (size_t i = 0; i < weights.size(); i++) {
			sum += *(weights[i]) * inputVector[i];
		}
		sum -= threshold;
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
	INeuron * NeuronOfRNN::clone()
	{
		return new NeuronOfRNN(*this);
	}
}