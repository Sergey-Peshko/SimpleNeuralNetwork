#pragma once
#include "stdafx.h"
#include "DataItem.h"
namespace neuralNet {
	template<class T>
	class ILearningStrategy {
		virtual void train(T network, (vector<DataItem<float>>* data) = 0;
	};
}
