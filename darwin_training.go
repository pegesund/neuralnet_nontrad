package main

import (
	"fmt"
	"math/rand"
	"sync/atomic"
)

/*
	Train by creating hierarchies of competing nets
    Only the "best" of these survives and get promoted to alhpa nets

*/

var winners uint64 = 0
var treeJobs = make(chan *training, 100)
var results = make(chan *training, 100)

// pick training from jobs and pass on to train function
func trainTreeWorker() {
	for training := range treeJobs {

		fmt.Println(training)
	}
}

// calculates average error
func averageErrorInNet(set *trainingSet, net *net, oldBest float64) float64 {
	sumErr := 0.0
	for i := 0; i < len(set.in); i++ {
		setInput(net, set.in[i])
		updateValues(net)
		sumErr += calcError(net, set.out[i]) / float64(len(set.in))
		if sumErr > oldBest {
			break
		}
	}
	net.error = sumErr
	return sumErr
}

// creates clones of a net and keeps the winner
// scores agains the hole traning net
// returns the winning net
func createCloneMutateAndEvaluate(net *net, training *training) *net {
	winner := &net
	updateValues(net)
	netAvgErr := averageErrorInNet(training.tSet, net, 1000)
	for i := 0; i < training.swapInter && i < 100000; i++ {
		if i%1000 == 0 {
			fmt.Println("Counting: ", i)
		}
		clone := cloneNet(net)
		permuteNet(clone)
		updateValues(clone)
		cloneAvgErr := averageErrorInNet(training.tSet, clone, netAvgErr)
		if cloneAvgErr < netAvgErr {
			winner = &clone
			netAvgErr = cloneAvgErr
			fmt.Printf("%d New winner %f \n: ", i, cloneAvgErr)
			atomic.AddUint64(&winners, 1)
		}
	}
	netPtr := &net
	*netPtr = *winner
	return *winner
}

// permute the net
// changes only the weights and directions
func permuteNet(net *net) {
	treeSize := len(net.layers)
	for i := 0; i < treeSize; i++ {
		layer := &layer{make([]neuron, len(net.layers[i].neurons))}
		for j := 0; j < len(layer.neurons); j++ {
			neurone := &net.layers[i].neurons[j]
			synapses := &neurone.synapses
			for k := 0; k < len(*synapses); k++ {
				synapse := &(*synapses)[k]
				if rand.Intn(10) == 1 {
					synapse.direction = synapse.direction * -1
				}
				if rand.Intn(10) == 1 {
					synapse.incSize = rand.Float64()
				} else {
					synapse.incSize = synapse.incSize + (synapse.direction * (synapse.incSize / 20))
				}
				synapse.weight += (synapse.incSize + 1) * synapse.direction * (1 + net.mutationInc)
			}
		}
	}
}

// spawn number of threads to do training in
func train(training *training) {
	for w := 1; w <= training.threads; w++ {
		go trainTreeWorker()
	}
}

// creates childrens recursively
// for each depth minimize the mutationInc and lower the genetic difference

func createNetChildren(depth int, currentDepth int, nets []*net, diversity int) {
	for i := 0; i < len(nets); i++ {
		children := make([]*net, diversity)
		for j := 0; j < diversity; j++ {
			clone := cloneNet(nets[i])
			clone.mutationInc = nets[i].mutationInc * 0.5
			permuteNet(clone)
			children[j] = clone
		}
		nets[i].children = children
		if currentDepth+1 < depth {
			createNetChildren(depth, currentDepth+1, children, diversity)
		}
	}
}

func createWood(diversity int, layers []int, bias float64) *wood {
	nets := make([]*net, diversity)
	for i := 0; i < len(nets); i++ {
		nets[i] = initRandom(layers[:], bias, nil)
	}
	return &wood{nets, diversity}
}

func main2() {
	fmt.Println("HUPP!!!")
	// layers := []int{3, 2}
}
