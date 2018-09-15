package main

import (
	"fmt"
	"github.com/kr/pretty"
	"math"
	"math/rand"
	"runtime"
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
		fmt.Print(i, " - ", n.val, " ,  ")
	}
	fmt.Println()
	for i, n := range net.layers[netSize-1].neurons {
		fmt.Print(i, " - ", n.val, " ,  ")
	}
}

// creates and initiates net with random values
func initRandom(layers []int, bias float64) *net {
	var net = net{make([]layer, len(layers)), bias, 1}
	for i := 0; i < len(layers); i++ {
		layerLen := layers[i]
		layer := layer{make([]neuron, layerLen)}
		for j, _ := range layer.neurons {
			layer.neurons[j].val = 0
			if i < len(net.layers)-1 {
				fmt.Println(i)
				synapses := make([]synapse, layers[i+1])
				for k, _ := range synapses {
					synapses[k].weight = randomVal()
					synapses[k].incSize = 1 + (randomVal() / 10)
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

// clones a neural network

func cloneNet(oldNet *net) *net {
	treeSize := len(oldNet.layers)
	var newNet = net{make([]layer, treeSize), oldNet.bias, oldNet.mutationInc}
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
			newNeuron := neuron{oldNeurone.val, newSynapses}
			layer.neurons[j] = newNeuron
		}
	}
	return &newNet
}

// activation function is play sigmoid
// try out: https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

func activateSigmoid(val float64) float64 {
	return 1 / (1 + math.Exp(-val))
}

// minimize this function from movement to movement

func calcError(net *net, expectedResult []float64) float64 {
	netSize := len(net.layers)
	var e float64 = 0
	for i := 0; i < len(net.layers[netSize-1].neurons); i++ {
		e += math.Pow(expectedResult[i]-net.layers[0].neurons[i].val, 2)
	}
	return e
}

// set values in input neurons

func setInput(net *net, input []float64) {
	for i := 0; i < len(input); i++ {
		net.layers[0].neurons[i].val = input[i]
	}
}

// update valus in higher layers based on weight
// standard feed forward

func updateValues(net *net) {
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
				sum += neuronBelow.synapses[j].weight * neuronBelow.val
			}
			net.layers[i].neurons[j].val = activateSigmoid(sum)
		}
	}
	seeNet(*net)
	fmt.Println("Updated network")
}


func benchmarkClone(net *net) {
	start := time.Now()
	for i := 0; i < 10000000; i++ {
		t2 := cloneNet(net)
		if (t2.bias ==3 ) { fmt.Println("OJ")}
	}
	elapsed := time.Since(start)
	fmt.Printf("\n Time took %s", elapsed)
	fmt.Println(runtime.NumCPU())
}

func main() {
	layers := []int{3, 2}
	mynet := initRandom(layers[:], 0)
	setInput(mynet, []float64{1, 2, 3})
	updateValues(mynet)
	seeNet(*mynet)
	fmt.Println("--------------")
	adjustNet(mynet)
	seeNet(*mynet)
	// seeInputOutput(*mynet)
}

