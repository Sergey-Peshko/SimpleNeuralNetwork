#pragma once
#include "..\stdafx.h"
#include "..\Data\DataItem.h"

namespace neuralNet {
	class INeuralNetwork {
	public:
		virtual vector<float> calculateOutput(vector<float> inputVector) = 0;
		virtual void save(std::string way) = 0;
		virtual void train(vector<DataItem<float>>& data) = 0;
	};
}