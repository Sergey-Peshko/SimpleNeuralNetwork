#pragma once
#include "..\stdafx.h"
template<class T>
class IOutputInterpretatorLogic {
public:
	virtual bool compare(vector<T>& currOutput, vector<T>& etalonOutput) = 0;
};