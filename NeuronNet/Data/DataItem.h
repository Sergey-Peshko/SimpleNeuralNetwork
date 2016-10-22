#pragma once
#include "..\stdafx.h"

namespace neuralNet {
	template<typename T>
	class DataItem {
	private:
		vector<T> _input;
		vector<T> _output;
	public:
		DataItem() {
			
		}
		DataItem(const vector<T>& input, const vector<T>& output) {
			_input = input;
			_output = output;
		}
		void setInput(const vector<T>& value) {
			_input = value;
		}
		vector<T>& Input() {
			return _input;
		}
		void setOutput(const vector<T>& value) {
			_output = value;
		}
		vector<T>& Output() {
			return _output;
		}
	};
}