package main

import (
	"fmt"
	"github.com/kr/pretty"
	"math"
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
	return rand.Float64()
}

// Function to see values in first and last layer
func seeInputOutput(net net) {
	netSize := len(net.layers)
	for i, n := range net.layers[0].neurons {
		fmt.Print(i, " - ", n.out, " ,  ")
	}
	fmt.Println()
	for i, n := range net.layers[netSize-1].neurons {
		fmt.Print(i, " - ", n.out, " ,  ")
	}
}

// creates and initiates net with random values
func initRandom(layersLen []int, bias float64, layersActivate []func(float64) float64, layersActivateVals []ActivationFunction) *net {
	var net = net{make([]layer, len(layersLen)), bias, 1,
		layersLen, layersActivate, layersActivateVals, 0, 0, ClonedNet}
	for i := 0; i < len(layersLen); i++ {
		layerLen := layersLen[i]
		layer := layer{make([]neuron, layerLen)}
		for j, _ := range layer.neurons {
			layer.neurons[j].out = 0
			layer.neurons[j].in = 0
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
	setInput(net, input)
	feedForward(net)
	fmt.Println("----")
	seeInputOutput(*net)
	fmt.Println("====")
	fmt.Println("Error: ", net.error)
}

// clones a neural network

func cloneNet(oldNet *net) *net {
	atomic.AddUint64(&cloneCounter, 1)
	treeSize := len(oldNet.layers)
	var newNet = net{make([]layer, treeSize), oldNet.bias, oldNet.mutationInc, oldNet.layersLength,
		oldNet.layersActivate, oldNet.layersActVal, 0, 0, oldNet.netType}
	for i := 0; i < treeSize; i++ {
		layer := layer{make([]neuron, len(oldNet.layers[i].neurons))}
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
		e += math.Pow(expectedResult[i]-net.layers[netSize-1].neurons[i].out, 2)
	}
	net.error = e
	return e
}

// set values in input neurons

func setInput(net *net, input []float64) {
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
			if i == 1 {
				sum = net.bias
			} else {
				sum = 0
			}
			// iterate layer below
			for k := 0; k < len(net.layers[i-1].neurons); k++ {
				neuronBelow := &net.layers[i-1].neurons[k]
				sum += neuronBelow.synapses[j].weight * neuronBelow.out
			}
			net.layers[i].neurons[j].in = sum
			// fmt.Println("I is: ", i,net.layersActivate[i] )
			net.layers[i].neurons[j].out = net.layersActivate[i](sum)
		}

		if net.layersActVal[i] == SoftMax {
			softMaxSum := 0.0
			for k := 0; k < len(net.layers[i].neurons); k++ {
				softMaxSum += net.layers[i].neurons[k].out
			}
			for k := 0; k < len(net.layers[i].neurons); k++ {
				net.layers[i].neurons[k].out = net.layers[i].neurons[k].out / softMaxSum
			}
		}
	}
	// seeNet(*net)
}

func benchmarkClone(net *net) {
	start := time.Now()
	for i := 0; i < 10000000; i++ {
		t2 := cloneNet(net)
		if t2.bias == 3 {
			fmt.Println("OJ")
		}
	}
	elapsed := time.Since(start)
	fmt.Printf("\n Time took %s", elapsed)
	fmt.Println(runtime.NumCPU())
}

func main3() {
	layersLength := []int{2, 3, 3, 3, 3, 1}
	layersActivate := []ActivationFunction{Identity, Tanh, Tanh, Tanh, Tanh, Tanh}
	wood := createWood(50, layersLength, 0.0, layersActivate)
	in := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	out := [][]float64{{0}, {1}, {1}, {0}}
	tSet := trainingSet{in, out}
	training := training{&tSet, 0, 10, 7, 300, 0.0, 200}
	trainWood(wood, &training)
}

func main() {
	fmt.Println("Ok, starting")
	start := time.Now()
	rand.Seed(time.Now().UTC().UnixNano())
	main3()
	// go func() { commands <- "hupp" }()
	// fmt.Println(<-commands)
	elapsed := time.Since(start)
	fmt.Println("Winners: ", winners)
	fmt.Println("CloneCounter: ", cloneCounter)
	fmt.Printf("\n Time took %s", elapsed)
}
