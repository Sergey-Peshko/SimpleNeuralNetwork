#pragma once
#include "..\..\stdafx.h"
#include "..\INeuralNetwork.h"
#include "..\..\Layers\ILayer.h"
#include "..\..\Layers\InvertedLayer.h"

namespace neuralNet {
	class IRecurentNeuralNetwork abstract : public INeuralNetwork {
	public:
		virtual vector<float> calculateInput(vector<float> outputVector) = 0;
		virtual InvertedLayer* getInvertedLayer() = 0;
	};
}