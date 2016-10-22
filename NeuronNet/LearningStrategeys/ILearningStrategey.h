#pragma once
#include "..\stdafx.h"
#include "..\Data\DataItem.h"

namespace neuralNet {
	template<class T>
	class ILearningStrategy {
	public:
		virtual void train(T* network, vector<DataItem<float>>& data) = 0;
	};
}
