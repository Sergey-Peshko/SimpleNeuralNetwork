#pragma once
#include "IOutputInterpretatorLogic.h"
#include "..\stdafx.h"
class MNISTInterpretatorLogic : public IOutputInterpretatorLogic<float> {
private:
	
public:
	MNISTInterpretatorLogic() {
		
	}
	// Унаследовано через IOutputInterpretator
	virtual bool compare(vector<float>& currOutput, vector<float>& etalonOutput) override;

};

bool MNISTInterpretatorLogic::compare(vector<float>& currOutput, vector<float>& etalonOutput) {
	if (currOutput.size() != currOutput.size()) {
		throw 2;
	}
	int max_index = 0;
	for (int i = 0; i < etalonOutput.size(); i++) {
		if (currOutput[i] > currOutput[max_index])
			max_index = i;
	}
	if (etalonOutput[max_index] == 1.)
		return true;
	else
		return false;
}