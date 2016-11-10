#pragma once
#include "..\..\stdafx.h"
#include "..\INeuralNetwork.h"
#include "..\..\Layers\ILayer.h"

namespace neuralNet {
	class IRecurentNeuralNetwork abstract : public INeuralNetwork {
	public:
		virtual vector<float> calculateInput(vector<float> outputVector) = 0;
	};
}