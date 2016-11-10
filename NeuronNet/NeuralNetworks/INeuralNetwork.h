#pragma once
#include "..\stdafx.h"
#include "..\Data\DataItem.h"
#include "..\Layers\ILayer.h"

namespace neuralNet {
	class INeuralNetwork {
	public:
		virtual vector<float> calculateOutput(vector<float> inputVector) = 0;
		virtual void save(std::string way) = 0;
		virtual void train(vector<DataItem<float>>& data) = 0;

		virtual ILayer* InutLayer() = 0;
		virtual ILayer* OutputLayer() = 0;
	};
}