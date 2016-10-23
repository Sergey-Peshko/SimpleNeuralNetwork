#pragma once
#include "..\stdafx.h"
#include "DataItem.h"

namespace neuralNet {
	class MNISTReader
	{
	private:
		int ReverseBytes(int v) // 32 bits = 4 bytes
		{
			// bit-manipulation version
			return (v & 0x000000FF) << 24 | (v & 0x0000FF00) << 8 |
				(v & 0x00FF0000) >> 8 | ((int)(v & 0xFF000000)) >> 24;
		}
		int ReadInt32(ifstream& is)
		{
			char buff[4];
			is.read(buff, 4);
			int* lol = (int*)buff;
			return *lol;
		}
		int ReadByte(ifstream& is)
		{
			char buff[1];
			is.read(buff, 1);
			return buff[0];
		}

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
		vector<DataItem<float>> LoadData(string pixelFile, string labelFile, int numImages = 60000)
		{
			// Load MNIST training set of 60,000 images into memory

			vector<DataItem<float>> result(numImages);

			vector<byte> pixels(784);


			ifstream brImages(pixelFile, ios::binary);
			ifstream brLabels(labelFile, ios::binary);

			if (!brImages.is_open()) {
				std::cout << "lol read" << std::endl;
			}
			if (!brLabels.is_open()) {
				cout << "lol read2" << endl;
			}

			int magic1 = ReadInt32(brImages);// stored as Big Endian
			magic1 = ReverseBytes(magic1); // convert to Intel format

			int imageCount = ReadInt32(brImages);
			imageCount = ReverseBytes(imageCount);

			int numRows = ReadInt32(brImages);
			numRows = ReverseBytes(numRows);
			int numCols = ReadInt32(brImages);
			numCols = ReverseBytes(numCols);

			int magic2 = ReadInt32(brLabels);
			magic2 = ReverseBytes(magic2);

			int numLabels = ReadInt32(brLabels);
			numLabels = ReverseBytes(numLabels);

			// each image
			for (int di = 0; di < numImages; ++di)
			{
				for (int i = 0; i < 784; ++i) // get 28x28 pixel values
				{
					pixels[i] = ReadByte(brImages);
				}

				byte lbl = ReadByte(brLabels); // get the label
				DataItem<float> dImage(pixels, lbl);
				result[di] = dImage;
			} // each image

			brImages.close();
			brLabels.close();

			return result;
		} // LoadData


		MNISTReader()
		{
		}

		~MNISTReader()
		{
		}
	};
}

