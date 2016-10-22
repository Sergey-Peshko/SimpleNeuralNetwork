#pragma once
#include"..\stdafx.h"
#include"IActivationFunction.h"

namespace neuralNet {
	class Sigmoid : public IActivationFunction{
	public:
		Sigmoid();
		Sigmoid(const Sigmoid& object);

		// Унаследовано через IActivationFunction
		virtual float calculate(float S) override;
		virtual float calculateFirstDerivative(float S) override;
		virtual IActivationFunction * clone() override;
	};
	Sigmoid::Sigmoid() {}

	Sigmoid::Sigmoid(const Sigmoid & object)
	{
	}

	float Sigmoid::calculate(float S)
	{
		return 1.f / (1.f + expf(-S));
	}

	float Sigmoid::calculateFirstDerivative(float S)
	{
		float y = calculate(S);
		return y * (1 - y);
	}

	IActivationFunction * Sigmoid::clone()
	{
		return new Sigmoid(*this);
	}
}