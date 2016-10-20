#pragma once
#include "stdafx.h"
#include "INeuralNetwork.h"
#include "ILayer.h"
namespace neuralNet {
	using std::vector;
	class IMultilayerNeuralNetwork : INeuralNetwork {
		virtual vector<ILayer*>* getLayers() = 0;
	};
}