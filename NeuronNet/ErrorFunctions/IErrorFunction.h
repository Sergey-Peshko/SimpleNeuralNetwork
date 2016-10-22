#pragma once
#include"..\stdafx.h"

namespace neuralNet {
	template<typename T>
	class IErrorFunction abstract{
	public:
		virtual float calculate(const vector<T>& etalon, const vector<T>& outOfNet) = 0;
		virtual T calculatePartialDerivaitve(const vector<T>& etalon, const vector<T>& outOfNet, size_t indexOfOutOfNet) = 0;
	};
}