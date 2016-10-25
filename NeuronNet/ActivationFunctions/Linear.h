#pragma once
#include"IActivationFunction.h"
namespace neuralNet {
	class Linear : public IActivationFunction
	{
	public:

		Linear();
		Linear(const Linear& object);

		// Унаследовано через IActivationFunction
		virtual float calculate(float S) override;
		virtual float calculateFirstDerivative(float S) override;
		virtual IActivationFunction * clone() override;
	};
	Linear::Linear()
	{
	}

	Linear::Linear(const Linear & object)
	{
	}

	float Linear::calculate(float S)
	{
		return S;
	}

	float Linear::calculateFirstDerivative(float S)
	{
		return 1;
	}

	IActivationFunction * Linear::clone()
	{
		return new Linear(*this);
	}
}