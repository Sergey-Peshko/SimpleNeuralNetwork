#pragma once
#pragma once
#include "..\stdafx.h"
#include "DataItem.h"

namespace neuralNet {
	class XORReader
	{
	private:
		template <typename T>
		std::string toString(T val)
		{
			ostringstream oss;
			oss << val;
			return oss.str();
		}

		template<typename T>
		T fromString(const std::string& s)
		{
			istringstream iss(s);
			T res;
			iss >> res;
			return res;
		}
	public:
		vector<DataItem<float>> LoadData(string xorFile, int numImages)
		{
			vector<DataItem<float>> result(numImages);

			ifstream file(xorFile);

			if (!file.is_open()) {
				std::cout << "lol read" << std::endl;
			}


			int imageCount; 
			file >> imageCount;
			int bitOf;
			file >> bitOf;
			
			vector<float> number(bitOf);

			// each image
			for (int di = 0; di < numImages; ++di)
			{
				string strNumber;
				file >> strNumber;
				int output;
				file >> output;

				DataItem<float> dImage(strNumber, output);
				result[di] = dImage;
			} // each image

			file.close();

			return result;
		} // LoadData
	};
}

