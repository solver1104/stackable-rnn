// TODO:
// - Implement AdaGrad Gradient Descent
// - Implement Gradient Clipping
// - Enable MiniBatch learning
// - Build regular Neural Network output layer

//#pragma warning( disable : 4996 )
#include <bits/stdc++.h>

#pragma GCC target ("avx2")
#pragma GCC optimization ("O3")
#pragma GCC optimization ("unroll-loops")

using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> distr(-1, 1);

vector<double> vectorAdd(vector<double> a, vector<double> b) {
	vector<double> sum;

	for (int i = 0; i < a.size(); i++) {
		sum.push_back(a[i] + b[i]);
	}

	return sum;
}

vector<double> scalarVectorProd(double a, vector<double> b) {
	vector<double> prod;

	for (int i = 0; i < b.size(); i++) {
		prod.push_back(a * b[i]);
	}

	return prod;
}

vector<double> activationFunc(vector<double> a) {
	vector<double> res;

	for (int i = 0; i < a.size(); i++) {
		res.push_back(1 / (1 + exp(-a[i] / 10)));
	}

	return res;
}

double derivActivationFunc(double a) {
	return a * (1 - a) / 10;
}

class RNNLayer {
	public:
		RNNLayer(int inputNodeCount, int outputNodeCount, double learnRate) {
			this->inputNodes = inputNodeCount;
			this->outputNodes = outputNodeCount;
			this->depth = 0;
			this->learningRate = learnRate;
			memoryCells.push_back({});

			for (int i = 0; i < outputNodeCount; i++) {
				memoryCells[0].push_back(0);
			}

			for (int i = 0; i < inputNodeCount; i++) {
				inputEdges.push_back({});
				for (int j = 0; j < outputNodeCount; j ++) {
					inputEdges[i].push_back(distr(gen));
				}
			}

			for (int i = 0; i < outputNodeCount; i++) {
				memoryEdges.push_back({});
				memoryBiases.push_back(distr(gen));
				for (int j = 0; j < outputNodeCount; j++) {
					memoryEdges[i].push_back(distr(gen));
				}
			}
		};

		vector<double> forwardprop(vector<double> input) {
			depth++;
			inputs.push_back(input);
			memoryCells.push_back({});

			for (int i = 0; i < outputNodes; i++) {
				memoryCells[depth].push_back(memoryBiases[i]);
			}

			for (int i = 0; i < inputNodes; i++) {
				memoryCells[depth] = vectorAdd(memoryCells[depth], scalarVectorProd(input[i], inputEdges[i]));
			}

			for (int i = 0; i < outputNodes; i++) {
				memoryCells[depth] = vectorAdd(memoryCells[depth], scalarVectorProd(memoryCells[depth - 1][i], memoryEdges[i]));
			}

			memoryCells[depth] = activationFunc(memoryCells[depth]);

			return memoryCells[depth];
		}

		vector<vector<double>> backprop(vector<vector<double>> costGradientOutputs) {
			vector<vector<double>> inputEdgeGradients;
			vector<vector<double>> memoryEdgeGradients;
			vector<double> memoryBiasGradients;
			vector<vector<double>> updatedCostGradientOutputs;
			vector<vector<double>> costGradientInputs;

			// Update partial derivatives of cost function with respect to output nodes to account for connections (they switch from being unconnected input nodes of the previous layer to connected memory nodes of the current layer)
			for (int i = 0; i <= depth; i++) {
				updatedCostGradientOutputs.push_back({});
			}
			
			for (int i = depth; i > 0; i--) {
				for (int j = 0; j < outputNodes; j++) {
					updatedCostGradientOutputs[i].push_back(costGradientOutputs[i][j]);
					if (i != depth) {
						for (int k = 0; k < outputNodes; k++) {
							updatedCostGradientOutputs[i][j] += updatedCostGradientOutputs[i + 1][k] * memoryEdges[j][k] * derivActivationFunc(memoryCells[i + 1][k]);
						}
					}
				}
			}

			// Calculate partial derivatives of cost function with respect to input nodes
			costGradientInputs.push_back({});

			for (int i = 1; i <= depth; i++) {
				costGradientInputs.push_back({});
				for (int j = 0; j < inputNodes; j++) {
					costGradientInputs[i].push_back(0);
					for (int k = 0; k < outputNodes; k++) {
						costGradientInputs[i][j] += inputEdges[j][k] * updatedCostGradientOutputs[i][k] * derivActivationFunc(memoryCells[i][k]);
					}
				}
			}

			// Gradient Descent to update Memory Edges
			// Calculate gradient of cost function with respect to each memory edge
			for (int i = 0; i < outputNodes; i++) {
				memoryEdgeGradients.push_back({});
				for (int j = 0; j < outputNodes; j++) {
					memoryEdgeGradients[i].push_back(0);

					for (int k = 1; k <= depth; k++) {
						memoryEdgeGradients[i][j] += memoryCells[k - 1][i] * updatedCostGradientOutputs[k][j] * derivActivationFunc(memoryCells[k][j]);
					}
				}
			}

			// Update memory edges using Gradient Descent
			for (int i = 0; i < outputNodes; i++) {
				for (int j = 0; j < outputNodes; j++) {
					memoryEdges[i][j] -= learningRate * memoryEdgeGradients[i][j];
				}
			}

			// Gradient Descent to update Input Edges
			// Calculate gradient of cost function with respect to each input edge
			for (int i = 0; i < inputNodes; i++) {
				inputEdgeGradients.push_back({});
				for (int j = 0; j < outputNodes; j++) {
					inputEdgeGradients[i].push_back(0);

					for (int k = 0; k < depth; k++) {
						inputEdgeGradients[i][j] += inputs[k][i] * updatedCostGradientOutputs[k + 1][j] * derivActivationFunc(memoryCells[k + 1][j]);
					}
				}
			}

			// Update input edges using Gradient Descent
			for (int i = 0; i < inputNodes; i++) {
				for (int j = 0; j < outputNodes; j++) {
					inputEdges[i][j] -= learningRate * inputEdgeGradients[i][j];
				}
			}

			// Gradient Descent to update Memory Biases
			// Calculate gradient of cost function with respect to each memory bias
			for (int i = 0; i < outputNodes; i++) {
				memoryBiasGradients.push_back(0);
				for (int j = 1; j <= depth; j++) {
					memoryBiasGradients[i] += updatedCostGradientOutputs[j][i] * derivActivationFunc(memoryCells[j][i]);
				}
			}

			// Update memory biases using Gradient Descent
			for (int i = 0; i < outputNodes; i++) {
				memoryBiases[i] -= learningRate * memoryBiasGradients[i];
			}

			// Return partial derivatives of cost function with respect to input nodes so the next stacked layer can use it
			return costGradientInputs;
		}

		void clearRNN() {
			depth = 0;	
			memoryCells.clear();
			inputs.clear();
			memoryCells.push_back({});

			for (int i = 0; i < outputNodes; i++) {
				memoryCells[0].push_back(0);
			}
		}
	
		int inputNodes;
		int outputNodes;
		int depth;
		double learningRate;
		vector<vector<double>> memoryCells;
		vector<vector<double>> inputs;
		vector<vector<double>> inputEdges;
		vector<vector<double>> memoryEdges;
		vector<double> memoryBiases;
};

vector<vector<double>> calcCostGradients(vector<vector<double>> pred, vector<vector<double>> goal) {
	vector<vector<double>> gradients;

	for (int i = 1; i <= pred.size(); i++) {
		gradients.push_back({});
		for (int j = 0; j < pred[i - 1].size(); j++) {
			gradients[i - 1].push_back(2 * pred[i - 1][j] - 2 * goal[i - 1][j]);
		}
	}

	return gradients;
}

const int numTrainingBatches = 1000;
const int numEpochs = 5;
vector<vector<double>> trainingDataAnswers;
vector<vector<double>> currentPred;

int main()
{
	/*
	string problemName = "names";
	ifstream cin(problemName + ".in");
	ofstream cout(problemName + ".out");

	ios::sync_with_stdio(0);
	cin.tie(0);


	RNNLayer RNNLayer1(26, 100, 1);
	RNNLayer RNNLayer2(100, 26, 1);

	string currentName;
	char currentChar;

	for (int cycle = 0; cycle < numEpochs; cycle++) {
		for (int i = 0; i < numTrainingBatches; i++) {
			cin >> currentName;
			trainingDataAnswers.clear();
			trainingDataAnswers.push_back({});
			currentPred.clear();
			currentPred.push_back({});
			RNNLayer1.clearRNN();
			RNNLayer2.clearRNN();
			for (int j = 0; j < 26; j++) {
				trainingDataAnswers[0].push_back(0);
				currentPred[0].push_back(0);
			}
			for (char x : currentName) {
				currentChar = tolower(x);
				trainingDataAnswers.push_back({});

				for (int j = 0; j < 26; j++) {
					trainingDataAnswers[trainingDataAnswers.size() - 1].push_back(0);
					if (currentChar - 97 == j) {
						trainingDataAnswers[trainingDataAnswers.size() - 1][j] = 1;
						currentPred[0][j] = 1;
					}
				}

				currentPred.push_back(RNNLayer2.forwardprop(RNNLayer1.forwardprop(trainingDataAnswers[trainingDataAnswers.size() - 1])));
			}

			RNNLayer1.backprop(RNNLayer2.backprop(calcCostGradients(currentPred, trainingDataAnswers)));
		}
	}
	cout << "aff";
	*/
	
	RNNLayer RNNLayer1(2, 2, 0.1);
	vector<vector<double>> pred;
	vector<vector<double>> ans;

	for (int i = 0; i < 500; i++) {
		pred.clear();
		ans.clear();
		RNNLayer1.clearRNN();
		pred.push_back({ sin(i / 10), sin(i / 5)});
		ans.push_back({ sin(i / 10), sin(i / 5)});
		for (int j = 0; j < 2; j++) {
			ans.push_back({ sin((i + j) / 10.0), sin((i + j) / 5.0)});
			pred.push_back(RNNLayer1.forwardprop({ sin((i + j - 1) / 10.0), sin((i + j - 1) / 5.0) }));
		}
		RNNLayer1.backprop(calcCostGradients(pred, ans));
	}

	cout << "asdf";
}