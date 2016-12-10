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
		
		DataItem(vector<float> pixels, byte label)
		{
			this->_input.resize(pixels.size());
			for (int i = 0; i < pixels.size(); i++) {
				this->_input[i] = pixels[i];
			}
			this-> _output.resize(10);
			for (int i = 0; i < 10; i++) {
				this->_output[i] = 0.;
			}
			this->_output[(int)label] = 1.;
		}
		DataItem(string input, int output)
		{
			this->_input.resize(input.size());
			for (int i = 0; i < input.size(); i++) {
				if(input[i]=='1')
					this->_input[i] = 1.;
				else
					this->_input[i] = 0.;
			}
			this->_output.resize(1);
			this->_output[0] = (float)output;
		}
	};
}