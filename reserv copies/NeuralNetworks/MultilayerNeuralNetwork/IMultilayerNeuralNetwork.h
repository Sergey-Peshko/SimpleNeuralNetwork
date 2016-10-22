#pragma once
#include "stdafx.h"
#include "INeuralNetwork.h"
#include "ILayer.h"
namespace neuralNet {
	using std::vector;
	class IMultilayerNeuralNetwork abstract : public INeuralNetwork {
	public:
		virtual vector<ILayer*>& Layers() = 0;
	};
}