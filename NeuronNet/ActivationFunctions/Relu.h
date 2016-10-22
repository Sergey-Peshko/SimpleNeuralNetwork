#pragma once
#include"IActivationFunction.h"
namespace neuralNet {
	class Relu : public IActivationFunction
	{
	private:

		float k;

	public:

		Relu();
		Relu(float k);
		Relu(const Relu& object);

		// Унаследовано через IActivationFunction
		virtual float calculate(float S) override;
		virtual float calculateFirstDerivative(float S) override;
		virtual IActivationFunction * clone() override;
	};
	Relu::Relu()
	{
		k = (float)0.01;
	}

	Relu::Relu(float k)
	{
		this->k = k;
	}

	Relu::Relu(const Relu & object)
	{
		this->k = object.k;
	}

	float Relu::calculate(float S)
	{
		return S > 0 ? S : k * S;
	}

	float Relu::calculateFirstDerivative(float S)
	{
		return S > 0 ? 1 : k;
	}

	IActivationFunction * Relu::clone()
	{
		return new Relu(*this);
	}
}