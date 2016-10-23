#pragma once
#include <vector>
using std::vector;
typedef unsigned char byte;
class DigitImage
{
public:
	// an MNIST image of a '0' thru '9' digit
	vector<double> pixels; // 0(white) - 255(black)
	vector<double> label; // '0' - '9'
	DigitImage() {}
	DigitImage(vector<byte> pixels, byte label)
	{
		this->pixels.resize(pixels.size());
		for (int i = 0; i < pixels.size(); i++) {
			this->pixels[i] = pixels[i];
		}
		this->label.resize(10);
		for (int i = 0; i < 10; i++) {
			this->label[i] = 0.;
		}
		this->label[(int)label] = 1.;
	}
};

