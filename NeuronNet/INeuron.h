#pragma once
#include "stdafx.h"
#include "IActivationFunction.h"
namespace neuralNet {
	using std::vector;
	class INeuron abstract {
		virtual vector<float>* getWeights() = 0;
		virtual float getThreshold() = 0;
		virtual void setThreshold(float value) = 0;
		virtual float calculateSum(vector<float>* inputVector) = 0;
		virtual float activate(vector<float>* inputVector) = 0;
		virtual void setLastState(float value) = 0;
		virtual float getLastState() = 0;
		virtual void setLastSum(float value) = 0;
		virtual float getLastSum() = 0;
		virtual vector<INeuron*>* getChilds() = 0;
		virtual vector<INeuron*>* getParents() = 0;
		virtual void setActivationFunction(IActivationFunction* value) = 0;
		virtual IActivationFunction* getIActivationFunction() = 0;
		virtual void setLastError(float value) = 0;
		virtual float getLastError() = 0;
	};
}