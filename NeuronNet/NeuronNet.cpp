// NeuronNet.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"

#include"NeuralNetworks\MultilayerNeuralNetwork\MLP.h"
#include"LearningStrategeys\BackpropagationLearningAlgorithm.h";
#include"ActivationFunctions\Relu.h"
#include"ActivationFunctions\Sigmoid.h"
#include"ActivationFunctions\Linear.h"
#include"Data\DataItem.h"
#include "NeuralNetworks\RecurentNeuralNetwork\OLRNN.h"
#include"LearningStrategeys\ContrastiveDivergence.h";
#include"LearningStrategeys\Configs\ContrastiveDivergenceAlgorithmConfig.h"
#include "LearningStrategeys\RestrictedBoltzmannMachines.h"
#include"Data\MNISTReader.h"
#include"LearningStrategeys\Configs\BackpropagationLearningAlgorithmConfig.h"
#include"OutputInterpretators\XORInterpretatorLogic.h"
#include"OutputInterpretators\MNISTInterpretatorLogic.h"
#include"Data\XORReader.h"

#include <iomanip>
using namespace neuralNet;

void print(vector<float> v) {
	for (int i = 0; i < v.size(); i++) {
		std::cout << v[i] << " ";
	}
	std::cout << std::endl;
}

void show(DataItem<float> data) {
	for (int i = 0; i < 28; i++, cout << endl)
		for (int j = 0; j < 28; j++)
			if (data.Input()[i * 28 + j] != 0)
				cout << std::setw(4) << data.Input()[i * 28 + j];
			else
				cout << std::setw(4)<<" ";
	int label;
	for (int i = 0; i < data.Output().size(); i++)
		if (data.Output()[i] == 1.)
			label = i;
	cout << "Label: " << label << endl;
}

int main()
{
	vector<DataItem<float>> data;
	vector<DataItem<float>> test;
	XORReader rd;
	data = rd.LoadData("data.xor", 12372);
	test = rd.LoadData("test.xor", 5000);
	BackpropagationLearningAlgorithmConfig bpaXOR;
	bpaXOR.setBatchSize(1);
	bpaXOR.setLearningRate(0.1);
	bpaXOR.setMaxEpoches(50);
	bpaXOR.setMinError(0.000'01);
	bpaXOR.setMinErrorChange(0.000'000'000'001);
	bpaXOR.setRegulaizationFactor(0.);
	bpaXOR.setTestSetError(0.0);
	bpaXOR.setTestSet(test);
	bpaXOR.setOutputInterpretatorLogic(new XORInterpretatorLogic());
	bpaXOR.setErrorFunction(new HalfSquaredEuclidianDistance<float>());

	ContrastiveDivergenceAlgorithmConfig cdXOR;
	cdXOR.setErrorFunction(new HalfSquaredEuclidianDistance<float>());
	cdXOR.setK(1);
	cdXOR.setLearningRate(0.1);
	cdXOR.setMaxEpoches(50);
	cdXOR.setMinError(0.000'01);
	cdXOR.setMinErrorChange(0.000'000'000'001);

	MLP mlp(32, { 96, 64, 32, 21, 15, 7, 2 }, 1, new Relu(), new Sigmoid(),
		new BackpropagationLearningAlgorithm(bpaXOR)
		, new RestrictedBoltzmannMachines(cdXOR)
	);
	mlp.save("tmp.txt");

	mlp.open("tmp.txt");

	mlp.preTrain(data);
	mlp.train(data);

	cout << endl;
	//system("pause");
	mlp.open("tmp.txt");

	mlp.setLearningStrategy(new BackpropagationLearningAlgorithm(bpaXOR));

	mlp.train(data);
	

    return 0;
}


/*
DataItem<float> tmp;

tmp.Input() = { 0,0 };
tmp.Output() = { 0 };
data.push_back(tmp);
tmp.Input() = { 0,1 };
tmp.Output() = { 1 };
data.push_back(tmp);
tmp.Input() = { 1,0 };
tmp.Output() = { 1 };
data.push_back(tmp);
tmp.Input() = { 1,1 };
tmp.Output() = { 0 };
data.push_back(tmp);
*/
/*
OLRNN rnn(2, 2, new Relu(), new ContrastiveDivergence());
rnn.train(data);

for (int i = 0; i < data.size(); i++)
print(rnn.calculateInput(rnn.calculateOutput(data[i].Input())));
*/
/*
BackpropagationLearningAlgorithmConfig bpaXOR;
bpaXOR.setBatchSize(1);
bpaXOR.setLearningRate(0.1);
bpaXOR.setMaxEpoches(150'000);
bpaXOR.setMinError(0.000'01);
bpaXOR.setMinErrorChange(0.000'000'000'001);
bpaXOR.setRegulaizationFactor(0.);
bpaXOR.setTestSetError(0.0);
bpaXOR.setTestSet(data);
bpaXOR.setOutputInterpretatorLogic(new XORInterpretatorLogic());
bpaXOR.setErrorFunction(new HalfSquaredEuclidianDistance<float>());

ContrastiveDivergenceAlgorithmConfig cdXOR;
cdXOR.setErrorFunction(new HalfSquaredEuclidianDistance<float>());
cdXOR.setK(1);
cdXOR.setLearningRate(0.1);
cdXOR.setMaxEpoches(150'000);
cdXOR.setMinError(0.000'01);
cdXOR.setMinErrorChange(0.000'000'000'001);

MLP mlp(2, { 2 }, 1, new Relu(), new Sigmoid(),
new BackpropagationLearningAlgorithm(bpaXOR)
, new RestrictedBoltzmannMachines(cdXOR)
);
mlp.save("tmp.txt");

mlp.open("tmp.txt");

mlp.preTrain(data);
mlp.train(data);

print(mlp.calculateOutput({ 0,0 }));
print(mlp.calculateOutput({ 0,1 }));
print(mlp.calculateOutput({ 1,0 }));
print(mlp.calculateOutput({ 1,1 }));

cout << endl;
system("pause");
mlp.open("tmp.txt");

mlp.setLearningStrategy(new BackpropagationLearningAlgorithm(bpaXOR));

mlp.train(data);

print(mlp.calculateOutput({ 0,0 }));
print(mlp.calculateOutput({ 0,1 }));
print(mlp.calculateOutput({ 1,0 }));
print(mlp.calculateOutput({ 1,1 }));
*/
/*
MNISTReader rd;
data = rd.LoadData("train-images.idx3-ubyte", "train-labels.idx1-ubyte");

BackpropagationLearningAlgorithmConfig bpaMNIST;
bpaMNIST.setBatchSize(1);
bpaMNIST.setLearningRate(0.1);
bpaMNIST.setMaxEpoches(20);
bpaMNIST.setMinError(0.000'01);
bpaMNIST.setMinErrorChange(0.000'000'000'001);
bpaMNIST.setRegulaizationFactor(0.0);
bpaMNIST.setTestSetError(0.001);
vector<DataItem<float>> test = rd.LoadData("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 10'000);
bpaMNIST.setTestSet(test);
bpaMNIST.setOutputInterpretatorLogic(new MNISTInterpretatorLogic());
bpaMNIST.setErrorFunction(new HalfSquaredEuclidianDistance<float>());

ContrastiveDivergenceAlgorithmConfig cdMNIST;
cdMNIST.setErrorFunction(new HalfSquaredEuclidianDistance<float>());
cdMNIST.setK(1);
cdMNIST.setLearningRate(0.01);
cdMNIST.setMaxEpoches(50);
cdMNIST.setMinError(0.000'01);
cdMNIST.setMinErrorChange(0.000'000'000'001);

MLP mlp(784, { 392, 196, 98, 49, 25 }, 10, new Relu(), new Sigmoid(),
new BackpropagationLearningAlgorithm(bpaMNIST)
, new RestrictedBoltzmannMachines(cdMNIST)
);
mlp.save("source.txt");

mlp.open("source.txt");

mlp.preTrain(vector<DataItem<float>>(data.begin(), data.begin() + 600));
mlp.save("rezultpretrain.txt");
mlp.train(data);
mlp.save("rezult.txt");
*/
