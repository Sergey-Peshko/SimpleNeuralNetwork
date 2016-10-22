#pragma once
#include "..\stdafx.h"
#include "..\ActivationFunctions\IActivationFunction.h"

namespace neuralNet {
	class INeuron abstract {
	private:
		virtual float calculateSum(const vector<float>& inputVector) = 0;
	public:
		virtual vector<float>& Weights() = 0;
		virtual float& Threshold() = 0;
		virtual float activate(const vector<float>& inputVector) = 0;
		virtual float getLastState() = 0;
		virtual float getLastSum() = 0;
	//	virtual vector<INeuron*>& getChilds() = 0;
	//	virtual vector<INeuron*>& getParents() = 0;
		virtual IActivationFunction* ActivationFunction() = 0;
		virtual float& LastError() = 0;
	};
}