#pragma once
#include "stdafx.h"
#include "INeuron.h"
namespace neuralNet {
	using std::vector;
	class ILayer abstract {
		virtual vector<float>* calculate(vector<float>* inputVector) = 0;
		virtual vector<float>* getLastOutput() = 0;
		virtual vector<INeuron*>* getNeurons() = 0;
		virtual size_t getInputDimension() = 0;
	};
}