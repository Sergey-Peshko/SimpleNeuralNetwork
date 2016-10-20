#pragma once
#include "stdafx.h"

namespace neuralNet {
	using std::vector;
	template<class T>
	class DataItem {
	private:
		vector<T> _input;
		vector<T> _output;
	public:
		DataItem() {
			_input = nullptr;
			_output = nullptr;
		}
		DataItem(vector<T> input, vector<T> output) {
			_input = input;
			_output = output;
		}
		void setInput(vector<T> value) {
			_input = value;
		}
		vector<T> getInput() {
			return _input;
		}
		void setOutput(vector<T> value) {
			_output = value;
		}
		vector<T> getOutput() {
			return _output;
		}
	};
}