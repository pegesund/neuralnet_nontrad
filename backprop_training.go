package main

import (
	"fmt"
	"math/rand"
)

var bias = 2.0

func setErrorInLastLayer(net *net, tSet *trainingSet, tSetNumber int) {
	lastLayer := &net.layers[len(net.layers)-1]
	for i := 0; i < len(tSet.out[tSetNumber]); i++ {
		loss := tSet.out[tSetNumber][i] - lastLayer.neurons[i].out
		lastLayer.neurons[i].err = loss
	}
}

func backPropagate(net *net, tSet *trainingSet, tSetNumber int, alpha float64, momentum float64) {
	setInputFirstLayer(net, tSet.in[tSetNumber])
	feedForward(net)
	lastLayer := &net.layers[len(net.layers)-1]
	setErrorInLastLayer(net, tSet, tSetNumber)
	if lastLayer.activateVal == SoftMax {
		softMaxPrimeReal(lastLayer)
	} else {
		for i := 0; i < len(lastLayer.neurons); i++ {
			{
				lastLayer.neurons[i].err = lastLayer.activatePrime(lastLayer.neurons[i].out) * lastLayer.neurons[i].err
			}
		}
	}

	// calc and propagete error backwards from last layer
	// for each neurone visited update error
	// and calculate delta changes for every synapse going out from that neurone
	for i := len(net.layers) - 2; i >= 0; i-- {
		for j := 0; j < len(net.layers[i].neurons); j++ {
			neurone := &net.layers[i].neurons[j]
			sumErr := 0.0
			for k := 0; k < len(neurone.synapses); k++ {
				synapse := &net.layers[i].neurons[j].synapses[k]
				toNeurone := &net.layers[i+1].neurons[k]
				hiddenDelta := synapse.weight * toNeurone.err
				sumErr += hiddenDelta
				// update current synapse with wight from above
				change := toNeurone.err * neurone.out
				synapse.weight += (change * alpha) + (synapse.incSize * momentum)
				synapse.incSize = change
			}
			neurone.err = net.layers[i].activatePrime(neurone.out) * sumErr
		}
	}
}

func trainBackPropagate(net *net, tSet *trainingSet, alpha float64, iterations int, momemtum float64, printInfo bool) {
	tSetNumber := 0
	fmt.Println("Starting backprop")
	for i := 0; i <= iterations; i++ {
		if i%1000 == 0 && i > -1 && printInfo {
			fmt.Printf("Iteration: %d cost %.13f - %d \n", i, calcCrossEntropy(net, tSet, 50),
				correctNumberOfPredictions(net, tSet, -1))
		}
		backPropagate(net, tSet, tSetNumber, alpha, momemtum)
		// tSetNumber = (tSetNumber + 1) % len(tSet.in)
		tSetNumber = rand.Intn(len(tSet.in))
	}
}
