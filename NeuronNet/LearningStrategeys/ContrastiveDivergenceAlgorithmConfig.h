#pragma once
#include "..\ErrorFunctions\IErrorFunction.h"
#include "..\ErrorFunctions\HalfSquaredEuclidianDistance.h"
namespace neuralNet {
	class ContrastiveDivergenceAlgorithmConfig {
		float learningRate;
		int k;
		int maxEpoches;
		float minError;
		float minErrorChange;
		IErrorFunction<float>* errorFunction;
	public:
		ContrastiveDivergenceAlgorithmConfig()
		{
			learningRate = 0.1;
			//batchSize = -1;
			k = 1;
			maxEpoches = 50;
			minError = 0.00001;
			minErrorChange = 0.000'000'000'001;
			errorFunction = new HalfSquaredEuclidianDistance<float>();
		}
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