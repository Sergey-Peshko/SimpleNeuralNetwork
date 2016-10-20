#pragma once
#include "stdafx.h"
namespace neuralNet{
	class IActivationFunction abstract{
		virtual float calculate(float S) = 0;
		virtual float calculateFirstDerivative(float S) = 0;
	};
}
