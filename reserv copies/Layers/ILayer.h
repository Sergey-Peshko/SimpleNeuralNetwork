#pragma once
#include "stdafx.h"
#include "INeuron.h"
namespace neuralNet {
	using std::vector;
	class ILayer abstract {
	public:
		virtual const vector<float>& calculate(const vector<float>& inputVector) = 0;
		virtual const vector<float>& LastOutput() = 0;
		virtual vector<INeuron*>& Neurons() = 0;
		virtual size_t getInputDimension() = 0;
	};
}