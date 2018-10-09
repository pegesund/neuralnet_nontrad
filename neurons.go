package main

import (
	"fmt"
	"github.com/kr/pretty"
	"math/rand"
	"runtime"
	"sync/atomic"
	"time"
)

func seeNet(net net) {
	pretty.Println(net)
}

/*
	update neurone weights, directions and inc_size with random values
*/

func randomVal() float64 {
	return random(1, -1)
}

func random(a, b float64) float64 {
	return (b-a)*rand.Float64() + a
}

// Function to see values in first and last layer
func seeInputOutput(net net) {
	netSize := len(net.layers)
	fmt.Print("{")
	for _, n := range net.layers[0].neurons {
		fmt.Printf("%f ,", n.in)
	}
	fmt.Println("}")
	fmt.Print("{")
	for _, n := range net.layers[netSize-1].neurons {
		fmt.Printf("%f , ", n.out)
	}
	fmt.Println("}")

}

func getLayersActivateVal(net *net) []ActivationFunction {
	res := make([]ActivationFunction, len(net.layers))
	for i := 0; i < len(net.layers); i++ {
		res[i] = net.layers[i].activateVal
	}
	return res
}

func getActivationFunction(a ActivationFunction) (func(float64) float64, func(float64) float64) {
	switch a {
	case Identity:
		return activateIdentity, activateIdentity
	case Tanh:
		return activateTanh, activateTanhPrime
	case Sigmoid:
		return activateSigmoid, activateSigmoidPrime
	case SoftMax:
		return activateSoftMax, activateSoftMaxPrime
	}
	return nil, nil
}

// creates and initiates net with random values
// used as a starting point for darwin-nets and backprop-nets
// the direction training prop is not used in gradient descent, only by the darwing nets
func initRandom(layersLen []int, bias bool, layersActivateVals []ActivationFunction) *net {
	var net = net{make([]layer, len(layersLen)), bias, 1,
		layersLen, 0, 0, ClonedNet}
	for i := 0; i < len(layersLen); i++ {
		neuroneLen := layersLen[i]
		activate, activatePrime := getActivationFunction(layersActivateVals[i])
		var biasUnitAdd int
		if !bias || i == len(layersLen)-1 || i == 0 {
			biasUnitAdd = 0
		} else {
			biasUnitAdd = 1
		} // do not add bias in first or last layer
		layer := layer{make([]neuron, neuroneLen+biasUnitAdd),
			activate, layersActivateVals[i], activatePrime}
		for j, _ := range layer.neurons {
			if i < len(net.layers)-1 {
				synapses := make([]synapse, layersLen[i+1])
				for k, _ := range synapses {
					synapses[k].weight = randomVal()
					synapses[k].incSize = randomVal() / 10
					if rand.Intn(2) == 1 {
						synapses[k].direction = Increase
					} else {
						synapses[k].direction = Decrease
					}
				}
				layer.neurons[j].synapses = synapses
			} else {
				layer.neurons[j].synapses = make([]synapse, 0)
			}
		}
		net.layers[i] = layer
	}
	return &net
}

func predict(input []float64, net *net) {
	setInputFirstLayer(net, input)
	feedForward(net)
	fmt.Println("----")
	seeInputOutput(*net)
	fmt.Println("====")
	fmt.Println("Error: ", net.error)
}

// clones a neural network

func cloneNet(oldNet *net) *net {
	atomic.AddUint64(&cloneCounter, 1)
	layerSize := len(oldNet.layers)
	var newNet = net{make([]layer, layerSize), oldNet.bias, oldNet.mutationInc, oldNet.layersLength,
		0, 0, oldNet.netType}
	for i := 0; i < layerSize; i++ {
		layer := layer{make([]neuron, len(oldNet.layers[i].neurons)),
			oldNet.layers[i].activateFunc, oldNet.layers[i].activateVal,
			oldNet.layers[i].activatePrime}
		newNet.layers[i] = layer
		for j := 0; j < len(layer.neurons); j++ {
			oldNeurone := &oldNet.layers[i].neurons[j]
			oldSynapses := oldNeurone.synapses
			newSynapses := make([]synapse, len(oldSynapses))
			for k := 0; k < len(newSynapses); k++ {
				newSynapses[k].direction = oldSynapses[k].direction
				newSynapses[k].weight = oldSynapses[k].weight
				newSynapses[k].incSize = oldSynapses[k].incSize
			}
			newNeuron := neuron{newSynapses, oldNeurone.in, oldNeurone.out, 0}
			layer.neurons[j] = newNeuron
		}
	}
	return &newNet
}

// activation function is play sigmoid
// try out: https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

// minimize this function from movement to movement

func calcError(net *net, expectedResult []float64) float64 {
	netSize := len(net.layers)
	var e float64 = 0
	for i := 0; i < len(net.layers[netSize-1].neurons); i++ {
		e += calcLossSquared(net.layers[netSize-1].neurons[i].out, expectedResult[i])
	}
	net.error = e
	return e
}

// set values in input neurons

func setInputFirstLayer(net *net, input []float64) {
	for i := 0; i < len(input); i++ {
		net.layers[0].neurons[i].in = input[i]
		net.layers[0].neurons[i].out = input[i]
	}
}

// update values in higher layers based on weight
// standard feed forward

func feedForward(net *net) {
	netSize := len(net.layers)
	for i := 1; i < netSize; i++ {
		neurons := net.layers[i].neurons
		for j := 0; j < len(neurons); j++ {
			var sum float64
			// iterate layer below
			for k := 0; k < len(net.layers[i-1].neurons); k++ {
				neuronBelow := &net.layers[i-1].neurons[k]
				if j < len(neuronBelow.synapses) {
					sum += neuronBelow.synapses[j].weight * neuronBelow.out
				}
			}
			net.layers[i].neurons[j].in = sum
			// fmt.Println("I is: ", i,net.activateFunc[i] )
			// if net.layers[i].activateVal == SoftMax {
			// 	fmt.Println("Before activate: ", sum)
			// }
			net.layers[i].neurons[j].out = net.layers[i].activateFunc(sum)
		}

		if net.layers[i].activateVal == SoftMax {
			// fmt.Println("-- SOFTMAX")
			softMaxSum := 0.0
			for k := 0; k < len(net.layers[i].neurons); k++ {
				softMaxSum += net.layers[i].neurons[k].out
				// fmt.Println("Out: ", net.layers[i].neurons[k].out)
			}
			for k := 0; k < len(net.layers[i].neurons); k++ {
				net.layers[i].neurons[k].out = net.layers[i].neurons[k].out / softMaxSum
			}
			// seeInputOutput(*net)
		}
	}
	// seeNet(*net)
}

func benchmarkClone(net *net) {
	start := time.Now()
	for i := 0; i < 10000000; i++ {
		t2 := cloneNet(net)
		if t2.bias == false {
			fmt.Println("OJ")
		}
	}
	elapsed := time.Since(start)
	fmt.Printf("\n Time took %s", elapsed)
	fmt.Println(runtime.NumCPU())
}

func testDarwinWoodTraining() {
	layersLength := []int{2, 3, 3, 3, 3, 1}
	layersActivate := []ActivationFunction{Identity, Tanh, Tanh, Tanh, Tanh, Tanh}
	wood := createWood(50, layersLength, false, layersActivate)
	in := [][]float64{{0, 1}, {0, 1}, {1, 0}, {1, 0}}
	out := [][]float64{{0}, {0}, {1}, {1}}
	tSet := trainingSet{in, out}
	training := darwinTraining{&tSet, 0, 10, 7, 300, 0.0, 200}
	trainWood(wood, &training)
	fmt.Println("Winners: ", winners)
	fmt.Println("CloneCounter: ", cloneCounter)
}

func testBackPropTraining() {
	layersLength := []int{2, 2, 1}
	layersActivate := []ActivationFunction{Identity, Sigmoid, Sigmoid, Sigmoid, Sigmoid}
	in := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	out := [][]float64{{0}, {1}, {1}, {0}}
	tSet := trainingSet{in, out}
	net := initRandom(layersLength, true, layersActivate)
	trainBackPropagate(net, &tSet, 0.6, 200000, 0.4, true)
	seeNet(*net)
}

func main() {
	fmt.Println("Ok, starting")
	start := time.Now()
	// rand.Seed(time.Now().UTC().UnixNano())
	rand.Seed(0)
	// testDarwinWoodTraining()
	testBackPropTraining()
	elapsed := time.Since(start)
	fmt.Printf("\n Time took %s", elapsed)
}
