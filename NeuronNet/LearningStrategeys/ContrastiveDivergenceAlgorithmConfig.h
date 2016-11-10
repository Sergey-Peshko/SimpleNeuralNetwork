#pragma once
#include "..\ErrorFunctions\IErrorFunction.h"
namespace neuralNet {
	class ContrastiveDivergenceAlgorithmConfig {
		float learningRate;
		int k;
		int maxEpoches;
		float minError;
		float minErrorChange;
		IErrorFunction<float>* errorFunction;
	public:
		void setLearningRate(float value) {
			learningRate = value;
		}
		float getLearningRate() {
			return learningRate;
		}
		void setK(int value) {
			k = value;
		}
		int getK() {
			return k;
		}
		int getMaxEpoches() {
			return maxEpoches;
		}
		void setMaxEpoches(int value) {
			maxEpoches = value;
		}
		float getMinError() {
			return minError;
		}
		void setMinError(float value) {
			minError = value;
		}
		float getMinErrorChange() {
			return minErrorChange;
		}
		void setMinErrorChange(float value) {
			minErrorChange = value;
		}
		IErrorFunction<float>* ErrorFunction() {
			return errorFunction;
		}
	};
}