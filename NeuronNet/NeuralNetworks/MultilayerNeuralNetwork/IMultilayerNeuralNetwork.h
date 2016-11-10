#pragma once
#include "..\..\stdafx.h"
#include "..\INeuralNetwork.h"
#include "..\..\Layers\ILayer.h"

namespace neuralNet {
	class IMultilayerNeuralNetwork abstract : public INeuralNetwork {
	public:
		virtual vector<ILayer*>& HiddenLayers() = 0;
	};
}