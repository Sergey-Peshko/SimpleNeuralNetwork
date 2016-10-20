#pragma once
#include "stdafx.h"
#include "IErrorFunction.h"
namespace neuralNet {
	class LearningAlgorithmConfig {
	public:
		float getLearningRate();
		void setLearningRate(float value);
		int getBatchSize();
		void setBatchSize(int value);
		float getRegularizationFactor();
		void setRegulaizationFactor(float value);
		int getMaxEpoches();
		void setMaxEpoches(int value);
		float getMinError();
		void setMinError(float value);
		float getMinErrorChange();
		void setMinErrorChange(float value);
		IErrorFunction<float>* getErrorFunction();
		void setErrorFunction(IErrorFunction<float>* value);
	};
}