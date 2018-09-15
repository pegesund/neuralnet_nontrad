package main

import (
	"fmt"
	"math/rand"
)

/*
	Train by creating a serie of comepting nets
    only the "best" survives, but this tree extends functions from the siblings

*/

var treeJobs = make(chan *training, 100)
var results = make(chan *training, 100)

// pick training from jobs and pass on to train function
func trainTreeWorker() {
	for training := range treeJobs {

		fmt.Println(training)
	}
}

// permute the net
func permuteNet(net *net) {
	treeSize := len(net.layers)
	for i := 0; i < treeSize; i++ {
		layer := &layer{make([]neuron, len(net.layers[i].neurons))}
		for j := 0; j < len(layer.neurons); j++ {
			neurone := &net.layers[i].neurons[j]
			synapses := &neurone.synapses
			for k := 0; k < len(*synapses); k++ {
				synapse := &(*synapses)[k]
				if rand.Intn(3) == 1 {
					synapse.direction = synapse.direction * -1
				}
				synapse.incSize = rand.Float64() + 1
			}
		}
	}
}

// adjusts the weights based on data from the surviving nets
func adjustNetWeights(net *net) {
	treeSize := len(net.layers)
	for i := 0; i < treeSize; i++ {
		layer := layer{make([]neuron, len(net.layers[i].neurons))}
		for j := 0; j < len(layer.neurons); j++ {
			neurone := &net.layers[i].neurons[j]
			synapses := &neurone.synapses
			for k := 0; k < len(*synapses); k++ {
				synapse := &(*synapses)[k]
				synapse.weight = synapse.weight * net.mutationInc * synapse.incSize * synapse.direction
			}
		}
	}
}

// spawn number of threads to do training in
func train(training *training) {
	for w := 1; w <= training.threads; w++ {
		go trainWorker()
	}
}

func main2() {
	fmt.Println("HUPP!!!")

}
