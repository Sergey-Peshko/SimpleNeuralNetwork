#pragma once
#include "IOutputInterpretatorLogic.h"
#include "..\stdafx.h"
class XORInterpretatorLogic : public IOutputInterpretatorLogic<float> {
private:
	float threshold;
public:
	XORInterpretatorLogic() {
		threshold = 0.5;
	}
	// Унаследовано через IOutputInterpretator
	virtual bool compare(vector<float>& currOutput, vector<float>& etalonOutput) override;

};

bool XORInterpretatorLogic::compare(vector<float>& currOutput, vector<float>& etalonOutput) {
	if (currOutput.size() != etalonOutput.size())
		throw 2;
	//if (currOutput.size() % 2 != 0)
	//	throw 1;
	//vector<size_t> max_indaxes(currOutput.size() / 2);
	for (int i = 0; i < currOutput.size(); i++) {
		if (currOutput[i] > threshold && etalonOutput[i] == 1.)
			continue;
		else
			if (currOutput[i] < threshold && etalonOutput[i] == 0)
				continue;
			else
				return false;
	}
	return true;
}