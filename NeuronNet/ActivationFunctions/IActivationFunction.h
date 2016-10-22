#pragma once
namespace neuralNet{
	class IActivationFunction abstract{
	public:
		virtual float calculate(float S) = 0;
		virtual float calculateFirstDerivative(float S) = 0;
	};
}
