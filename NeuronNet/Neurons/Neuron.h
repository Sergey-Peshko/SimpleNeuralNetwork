#pragma once
#include"INeuron.h"
namespace neuralNet {
	class Neuron : public INeuron{
	private:
		 vector<float> weights;
		 float threshold;
		 float lastState;
		 float lastSum;
		 IActivationFunction* activationFunction;
		 float lastError;

		 virtual float calculateSum(const vector<float>& inputVector) override;
	public:
		Neuron(size_t weightsSize, 
			std::mt19937& generator, 
			std::uniform_real_distribution<float>& urd,
			IActivationFunction* activationFunction);
		~Neuron();
		// ������������ ����� INeuron
		virtual vector<float>& Weights() override;
		virtual float & Threshold() override;
		virtual float activate(const vector<float>& inputVector) override;
		virtual float getLastState() override;
		virtual float getLastSum() override;
		virtual IActivationFunction * ActivationFunction() override;
		virtual float & LastError() override;
	};
	Neuron::Neuron(size_t weightsSize,
		std::mt19937& generator,
		std::uniform_real_distribution<float>& urd,
		IActivationFunction* activationFunction) : weights(weightsSize)
	{
		for (size_t i = 0; i < weightsSize; i++) {
			weights[i] = urd(generator);
		}
		this->activationFunction = activationFunction->clone();
	}
	Neuron::~Neuron() {
		delete activationFunction;
	}

	vector<float>& Neuron::Weights()
	{
		return weights;
	}

	float & Neuron::Threshold()
	{
		return threshold;
	}

	float Neuron::calculateSum(const vector<float>& inputVector)
	{
		float sum = 0;
		for (size_t i = 0; i < weights.size(); i++) {
			sum += weights[i] * inputVector[i];
		}
		sum -= threshold;
		lastSum = sum;
		return sum;
	}

	float Neuron::activate(const vector<float>& inputVector)
	{
		lastState = activationFunction->calculate(calculateSum(inputVector));
		return lastState;
	}

	float Neuron::getLastState()
	{
		return lastState;
	}

	float Neuron::getLastSum()
	{
		return lastSum;
	}

	IActivationFunction * Neuron::ActivationFunction()
	{
		return activationFunction;
	}

	float & Neuron::LastError()
	{
		return lastError;
	}

}