#pragma once
#include "stdafx.h"
#include "IErrorFunction.h"
namespace neuralNet {
	template<typename T>
	class HalfSquaredEuclidianDistance : public IErrorFunction<T> {
	public:
		virtual float calculate(const vector<T>& etalon, const vector<T>& outOfNet) override;
		virtual T calculatePartialDerivaitve(const vector<T>& etalon, const vector<T>& outOfNet, size_t indexOfOutOfNet) override;
	};
	template<typename T>
	float HalfSquaredEuclidianDistance<T>::calculate(const vector<T>& etalon, const vector<T>& outOfNet) {
		double d = 0;
		for (int i = 0; i < v1.Length; i++)
		{
			d += (v1[i] - v2[i]) * (v1[i] - v2[i]);
		}
		return 0.5 * d;
	}
	template<typename T>
	T HalfSquaredEuclidianDistance<T>::calculatePartialDerivaitve(const vector<T>& etalon, const vector<T>& outOfNet, size_t indexOfOutOfNet) {
		return outOfNet[indexOfOutOfNet] - etalon[indexOfOutOfNet];
	}
}
