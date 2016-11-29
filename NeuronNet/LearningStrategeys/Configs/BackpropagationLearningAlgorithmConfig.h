#pragma once
#include "..\..\stdafx.h"
#include "..\..\ErrorFunctions\IErrorFunction.h"
#include "..\..\ErrorFunctions\HalfSquaredEuclidianDistance.h"
#include "..\..\OutputInterpretators\IOutputInterpretatorLogic.h"

namespace neuralNet {
	class BackpropagationLearningAlgorithmConfig {
	private:
		float learningRate;
		int batchSize;
		float regularizationFactor;
		int maxEpoches;
		float minError;
		float minErrorChange;
		vector<DataItem<float>> test;
		IErrorFunction<float>* errorFunction;
		IOutputInterpretatorLogic<float>* interpretator;
		float testSetError;
	public:
		BackpropagationLearningAlgorithmConfig() {
			learningRate = 0.1;
			batchSize = -1;
			regularizationFactor = 0.005;	//0.005
			maxEpoches = 150'000;
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
		float getTestSetError() {
			return testSetError;
		}
		void setTestSetError(float value) {
			testSetError = value;
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
		vector<DataItem<float>>& getTestSet() {
			return test;
		}
		void setTestSet(vector<DataItem<float>>& testset) {
			test = testset;
		}
		IErrorFunction<float>* getErrorFunction() {
			return errorFunction;
		}
		void setErrorFunction(IErrorFunction<float>* err) {
			errorFunction = err;
		}
		IOutputInterpretatorLogic<float>*  getOutputInterpretatorLogic() {
			return interpretator;
		}
		void setOutputInterpretatorLogic(IOutputInterpretatorLogic<float>* interpretator) {
			this->interpretator = interpretator;
		}
	};
}