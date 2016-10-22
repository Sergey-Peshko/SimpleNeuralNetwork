#pragma once
#include "..\stdafx.h"
#include "..\ErrorFunctions\IErrorFunction.h"
#include "..\ErrorFunctions\HalfSquaredEuclidianDistance.h"

namespace neuralNet {
	class LearningAlgorithmConfig {
	private:
		float learningRate;
		int batchSize;
		float regularizationFactor;
		int maxEpoches;
		float minError;
		float minErrorChange;
		IErrorFunction<float>* errorFunction;
	public:
		LearningAlgorithmConfig() {
			learningRate = 0.1;
			batchSize = -1;
			regularizationFactor = 0.5;
			maxEpoches = 1000;
			minError = 0.00001;
			minErrorChange = 0.000'000'000'001;
			errorFunction = new HalfSquaredEuclidianDistance<float>();
		}
		float getLearningRate() {
			return learningRate;
		}
		void setLearningRate(float value) {
			learningRate = value;
		}
		int getBatchSize() {
			return batchSize;
		}
		void setBatchSize(int value) {
			batchSize = value;
		}
		float getRegularizationFactor() {
			return regularizationFactor;
		}
		void setRegulaizationFactor(float value) {
			regularizationFactor = value;
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