package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
)

/*
	Train by creating hierarchies of competing nets
    Only the "best" of these survives and get promoted to alhpa nets

*/

var maxConcurrent = 100
var winners uint64
var cloneCounter uint64
var mutateJobs = make(chan *trainMsg)
var mutateResults = make(chan *trainMsg)
var wg sync.WaitGroup

// pick darwinTraining from jobs and pass on to train function
func trainNetWorker() {
	for {
		select {
		case tMsg := <-mutateJobs:
			tMsg.wood.nets[tMsg.netNumber] = createCloneMutateAndEvaluate(tMsg.wood.nets[tMsg.netNumber], tMsg.training)
			go func() { mutateResults <- tMsg }()
			wg.Done()
		}
	}
}

// calculates average error
func averageErrorInNet(set *trainingSet, net *net, oldBest float64) float64 {
	sumErr := 0.0
	for i := 0; i < len(set.in); i++ {
		setInputFirstLayer(net, set.in[i])
		feedForward(net)
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
func createCloneMutateAndEvaluate(net *net, training *darwinTraining) *net {
	winner := &net
	feedForward(net)
	netAvgErr := averageErrorInNet(training.tSet, net, 1000)
	for i := 0; i < training.cloneIterations && training.errPass < netAvgErr; i++ {
		if i%100 == 0 {
			// fmt.Println("Counting: ", i, " average-err: ", netAvgErr)
		}
		clone := cloneNet(net)
		permuteNet(clone)
		feedForward(clone)
		cloneAvgErr := averageErrorInNet(training.tSet, clone, netAvgErr)
		if cloneAvgErr < netAvgErr {
			winner = &clone
			netAvgErr = cloneAvgErr
			// fmt.Printf("%d New winner %f \n: ", i, cloneAvgErr)
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
	layerSize := len(net.layers)
	for i := 0; i < layerSize; i++ {
		layer := &layer{make([]neuron, len(net.layers[i].neurons)),
			net.layers[i].activateFunc,
			net.layers[i].activateVal,
			net.layers[i].activatePrime}
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

func sortNetsByErr(nets []*net) {
	sort.Slice(nets, func(i, j int) bool {
		return nets[i].error < nets[j].error
	})
}

func trainOneGeneration(training *darwinTraining, wood *wood) {
	wg.Add(len(wood.nets))
	for i := 0; i < len(wood.nets); i++ {
		mutateJobs <- &trainMsg{training, wood, i}
	}
	for i := 0; i < len(wood.nets); i++ {
		<-mutateResults
	}
	wg.Wait()
}

func createWood(diversity int, layers []int, bias bool, layersActivateVals []ActivationFunction) *wood {
	if diversity%2 != 0 {
		diversity++
	}
	nets := make([]*net, diversity)
	for i := 0; i < len(nets); i++ {
		nets[i] = initRandom(layers[:], bias, layersActivateVals)
	}
	return &wood{nets, diversity}
}

func woodTotalErr(wood *wood, training *darwinTraining) (total float64) {
	total = 0.0
	for i := 0; i < len(wood.nets); i++ {
		feedForward(wood.nets[i])
		e := averageErrorInNet(training.tSet, wood.nets[i], math.MaxFloat64)
		total += e
		fmt.Printf(" %.13f ( %d - %d )  ", e, wood.nets[i].generation, wood.nets[i].netType)
	}
	fmt.Printf(" - total: %.13f \n", total)
	return
}

func trainWood(wood *wood, training *darwinTraining) (bestNet *net) {
	// spawn number of threads to do darwinTraining in
	for w := 0; w < training.threads; w++ {
		go trainNetWorker()
	}
	woodTotalErr(wood, training)
	for i := 0; i < training.runGenerations; i++ {
		fmt.Println("----- Iteration: ", i)
		trainOneGeneration(training, wood)
		sortNetsByErr(wood.nets)

		netSize := len(wood.nets)
		for j := 0; j < netSize/2; j++ {
			if wood.nets[(netSize/2)+j].generation < training.minGenerations {
				continue
			}
			father := rand.Intn(netSize / 2)
			mother := rand.Intn(netSize / 2)
			for father == mother {
				mother = rand.Intn(netSize / 2)
			}
			if rand.Intn(2) == 1 {
				wood.nets[(netSize/2)+j] = mergeNetsRandom(wood.nets[mother], wood.nets[father])
			} else {
				n := (netSize / 2) + j
				layers := wood.nets[n].layers
				layersLen := make([]int, len(layers))
				for l := 0; l < len(layersLen); l++ {
					layersLen[l] = len(layers[l].neurons)
				}
				wood.nets[n] = initRandom(layersLen, wood.nets[n].bias, getLayersActivateVal(wood.nets[n]))
				wood.nets[n].netType = NewRandNet
			}
		}
		for j := 0; j < netSize; j++ {
			wood.nets[j].generation++
		}
	}
	woodTotalErr(wood, training)
	fmt.Println("Done with darwinTraining")
	return wood.nets[0]
}

func main2() {
	fmt.Println("HUPP!!!")
	// layers := []int{3, 2}
}
