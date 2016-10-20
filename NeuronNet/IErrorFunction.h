#pragma once
#include"stdafx.h"
namespace neuralNet {
	using std::vector;

	template<class T>
	class IErrorFunction abstract{
		virtual float calculate(vector<T>* etalon, vector<T>* outOfNet) = 0;
		virtual T calculatePartialDerivaitve(vector<T>* etalon, vector<T>* outOfNet, size_t indexOfOutOfNet) = 0;
	};
}