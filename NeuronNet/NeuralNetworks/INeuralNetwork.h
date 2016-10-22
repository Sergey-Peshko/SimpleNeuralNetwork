#pragma once
#include "..\stdafx.h"
#include "..\Data\DataItem.h"

namespace neuralNet {
	class INeuralNetwork {
	public:
		virtual vector<float> calculateOutput(vector<float>& inputVector) = 0;
		virtual void save(ostream& os) = 0;
		virtual void train(const vector<DataItem<float>>& data) = 0;
	};
}