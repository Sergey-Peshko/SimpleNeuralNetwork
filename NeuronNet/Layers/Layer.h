#pragma once
#include "ILayer.h"
#include"..\ActivationFunctions\IActivationFunction.h"
#include"..\Neurons\Neuron.h"
namespace neuralNet {
	class Layer : public ILayer {
	private:
		vector<INeuron*> neurons;
		vector<float> lastOut;
		size_t inputDimension;
	public:
		Layer(size_t inputDimension,size_t neuronsSize, IActivationFunction* activationFunction);
		Layer(vector<vector<float>> w, vector<float> thresholds, IActivationFunction* activationFunction);
		Layer(const Layer& obj);
		~Layer();
		// Унаследовано через ILayer
		virtual const vector<float>& calculate(const vector<float>& inputVector) override;
		virtual const vector<float>& LastOutput() override;
		virtual vector<INeuron*>& Neurons() override;
		virtual size_t getInputDimension() override;
		virtual ILayer* clone() override;
		virtual string toString() override;
	};
	Layer::Layer(size_t inputDimension, 
		size_t neuronsSize, 
		IActivationFunction* activationFunction) : 
		neurons(neuronsSize), 
		lastOut(neuronsSize),
		inputDimension(inputDimension)
	{
		std::mt19937 generator((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());
		float board = 1. / sqrtf(neuronsSize);
		//std::uniform_real_distribution<float> urd(-board, std::nextafter(board, FLT_MAX));
		std::uniform_real_distribution<float> urd(0, std::nextafter(board, FLT_MAX));
		//std::uniform_real_distribution<float> urd(0.4, 0.6);

		for (size_t i = 0; i < neuronsSize; i++) {
			neurons[i] = new Neuron(inputDimension, generator, urd, activationFunction);
		}
	}
	Layer::Layer(vector<vector<float>> w, vector<float> thresholds, IActivationFunction* activationFunction) :
		neurons(w.size()),
		lastOut(w.size())
	{
		if (!w.empty())
			inputDimension = w[0].size();
		else
			cout << "Error" << endl;

		for (int i = 0; i < w.size(); i++) {
			neurons[i] = new Neuron(w[i], thresholds[i], activationFunction);
		}
	}
	Layer::Layer(const Layer& obj) :
		inputDimension(obj.inputDimension),
		lastOut(obj.lastOut),
		neurons(obj.neurons.size())
	{
		for (size_t i = 0; i < neurons.size(); i++) {
			neurons[i] = obj.neurons[i]->clone();
		}
	}
	Layer::~Layer() {
		for (size_t i = 0; i < neurons.size(); i++) {
			delete neurons[i];
		}
	}
	const vector<float>& Layer::calculate(const vector<float>& inputVector)
	{
		for (size_t i = 0; i < neurons.size(); i++) {
			lastOut[i] = neurons[i]->activate(inputVector);
		}
		return lastOut;
	}

	const vector<float>& Layer::LastOutput()
	{
		return lastOut;
	}

	vector<INeuron*>& Layer::Neurons()
	{
		return neurons;
	}

	size_t Layer::getInputDimension()
	{
		return inputDimension;
	}
	ILayer* Layer::clone() {
		return new Layer(*this);
	}

	string Layer::toString() {
		ostringstream rez;
		rez.precision(32);
		rez << neurons.size() << ' '
			<< inputDimension << ' ';
		for (int i = 0; i < neurons.size(); i++) {
			for (int j = 0; j < inputDimension; j++) {
				rez << neurons[i]->Weights()[j] << ' ';
			}
			rez << neurons[i]->Threshold() << ' ';
		}

		return rez.str();
	}
}
