#pragma once
#include "stdafx.h"
#include "DataItem.h"
namespace neuralNet {
	using std::vector;
	using std::ostream;
	class INeuralNetwork {
		virtual vector<float>* calculateOutput(vector<float>* inputVector) = 0;
		virtual void save(ostream& os) = 0;
		virtual void train(vector<DataItem<float>>* data) = 0;
	};
}