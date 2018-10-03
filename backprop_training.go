package main

import "fmt"

/*
	sets the error value in neurones in last layer
*/
func calcErrorInLastLayer(net *net, tSet *trainingSet, tSetNumber int) {
	lastLayer := &net.layers[len(net.layers)-1]
	// fmt.Printf("Len last last layer: %d   len tset.out: %d: \n", len(lastLayer.neurons), len(tSet.out))
	for i := 0; i < len(tSet.out[tSetNumber]); i++ {
		lastLayer.neurons[i].err = calcLossSquared(lastLayer.neurons[i].out, tSet.out[tSetNumber][i])
	}
}

func backPropagate(net *net, tSet *trainingSet, tSetNumber int, alpha float64) {
	setInputFirstLayer(net, tSet.in[tSetNumber])
	feedForward(net)
	calcErrorInLastLayer(net, tSet, tSetNumber)

	// calc and propagete error backwards from last layer
	// for each neurone visited update error
	// and calculate delta changes for every synapse going out from that neurone
	for i := len(net.layers) - 2; i >= 0; i-- {
		for j := 0; j < len(net.layers[i].neurons); j++ {
			neurone := &net.layers[i].neurons[j]
			neurone.err = 0.0
			for k := 0; k < len(neurone.synapses); k++ {
				synapse := &net.layers[i].neurons[j].synapses[k]
				toNeurone := &net.layers[i+1].neurons[k]
				neurone.err += toNeurone.err * (synapse.weight / toNeurone.in)
				delta := -net.layers[i+1].activatePrime(toNeurone.in) * toNeurone.err * synapse.weight * alpha
				synapse.weight += delta
			}
		}
	}
}

func trainBackPropagate(net *net, tSet *trainingSet, alpha float64, iterations int) {
	tSetNumber := 0
	for i := 0; i < iterations; i++ {
		if i%10000 == 0 {
			seeInputOutput(*net)
			fmt.Printf("Iteration: %d \n", i)
		}
		backPropagate(net, tSet, tSetNumber, alpha)
		tSetNumber = (tSetNumber + 1) % (len(tSet.in) - 1)

	}
}
