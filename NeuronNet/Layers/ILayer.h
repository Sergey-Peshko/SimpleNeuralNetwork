#pragma once
#include "..\stdafx.h"
#include "..\Neurons\INeuron.h"

namespace neuralNet {
	class ILayer abstract {
	public:
		virtual const vector<float>& calculate(const vector<float>& inputVector) = 0;
		virtual const vector<float>& LastOutput() = 0;
		virtual vector<INeuron*>& Neurons() = 0;
		virtual size_t getInputDimension() = 0;
		virtual ILayer* clone() = 0;
		virtual string toString() = 0;
		virtual ~ILayer() {};
	};
}