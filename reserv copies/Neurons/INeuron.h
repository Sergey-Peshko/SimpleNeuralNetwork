#pragma once
#include "stdafx.h"
#include "IActivationFunction.h"
namespace neuralNet {
	using std::vector;
	class INeuron abstract {
	public:
		virtual vector<float>& Weights() = 0;
		virtual float& Threshold() = 0;
		virtual float calculateSum(vector<float>& inputVector) = 0;
		virtual float activate(vector<float>& inputVector) = 0;
		virtual float& LastState() = 0;
		virtual float& LastSum() = 0;
	//	virtual vector<INeuron*>& getChilds() = 0;
	//	virtual vector<INeuron*>& getParents() = 0;
		virtual IActivationFunction* ActivationFunction() = 0;
		virtual float& LastError() = 0;
	};
}