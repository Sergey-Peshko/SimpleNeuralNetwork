#pragma once
#include"INeuron.h"
#include "..\ActivationFunctions\Linear.h"
namespace neuralNet {
	class InputNeuron : public INeuron {
	private:
		vector<float> weights;
		float threshold;
		float lastState;
		float lastSum;
		IActivationFunction* activationFunction;
		float lastError;

		size_t numberInLayer;

		virtual float calculateSum(const vector<float>& inputVector) override;
	public:
		InputNeuron(size_t weightsSize,
			size_t numberInLayer);
		InputNeuron(const InputNeuron& obj);
		~InputNeuron();
		// Унаследовано через INeuron
		virtual vector<float>& Weights() override;
		virtual float & Threshold() override;
		virtual float activate(const vector<float>& inputVector) override;
		virtual float getLastState() override;
		virtual float getLastSum() override;
		virtual IActivationFunction * ActivationFunction() override;
		virtual float & LastError() override;
		virtual INeuron* clone() override;
	};
	InputNeuron::InputNeuron(size_t weightsSize,
		size_t numberInLayer) : 
		weights(weightsSize, 0), 
		threshold(0.0),
		numberInLayer(numberInLayer)
	{
		weights[numberInLayer] = 1.0f;
		this->activationFunction = new Linear();
	}
	InputNeuron::InputNeuron(const InputNeuron& obj) :
		weights(obj.weights),
		threshold(obj.threshold),
		lastState(obj.lastState),
		lastSum(obj.lastSum),
		lastError(obj.lastError),
		numberInLayer(obj.numberInLayer)
	{
		activationFunction = obj.activationFunction->clone();
	}
	InputNeuron::~InputNeuron() {
		delete activationFunction;
	}

	vector<float>& InputNeuron::Weights()
	{
		return weights;
	}

	float & InputNeuron::Threshold()
	{
		return threshold;
	}

	float InputNeuron::calculateSum(const vector<float>& inputVector)
	{
		lastSum = inputVector[numberInLayer] - threshold;
		return lastSum;
	}

	float InputNeuron::activate(const vector<float>& inputVector)
	{
		lastState = calculateSum(inputVector);
		return lastState;
	}

	float InputNeuron::getLastState()
	{
		return lastState;
	}

	float InputNeuron::getLastSum()
	{
		return lastSum;
	}

	IActivationFunction * InputNeuron::ActivationFunction()
	{
		return activationFunction;
	}

	float & InputNeuron::LastError()
	{
		return lastError;
	}
	INeuron* InputNeuron::clone() {
		return new InputNeuron(*this);
	}
}