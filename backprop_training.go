package main

import "fmt"

/*
	sets the error value in neurones in last layer
*/
func calcErrorInLastLayer2(net *net, tSet *trainingSet, tSetNumber int) {
	lastLayer := &net.layers[len(net.layers)-1]
	// fmt.Printf("Len last last layer: %d   len tset.out: %d: \n", len(lastLayer.neurons), len(tSet.out))
	for i := 0; i < len(tSet.out[tSetNumber]); i++ {
		loss := calcLossSquared(lastLayer.neurons[i].out, tSet.out[tSetNumber][i])
		fmt.Println("Loss: ", loss, " - training: ", tSet.out[tSetNumber][i])
		lastLayer.neurons[i].err = loss
	}
}

func setErrorInLastLayer(net *net, tSet *trainingSet, tSetNumber int) {
	lastLayer := &net.layers[len(net.layers)-1]
	for i := 0; i < len(tSet.out[tSetNumber]); i++ {
		loss := tSet.out[tSetNumber][i] - lastLayer.neurons[i].out
		lastLayer.neurons[i].err = loss
	}
}

func backPropagate(net *net, tSet *trainingSet, tSetNumber int, alpha float64) {
	setInputFirstLayer(net, tSet.in[tSetNumber])
	feedForward(net)
	lastLayer := &net.layers[len(net.layers)-1]
	setErrorInLastLayer(net, tSet, tSetNumber)
	for i := 0; i < len(lastLayer.neurons); i++ {
		lastLayer.neurons[i].err = lastLayer.activatePrime(lastLayer.neurons[i].out) * lastLayer.neurons[i].err
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
				synapse.weight += change * alpha
			}
			neurone.err = net.layers[i].activatePrime(neurone.out) * sumErr
		}
	}
}

func trainBackPropagate(net *net, tSet *trainingSet, alpha float64, iterations int, printInfo bool) {
	tSetNumber := 0
	for i := 1; i <= iterations; i++ {
		if printInfo && i%10000 == 0 {
			fmt.Printf("Iteration: %d cost %.13f \n", i, calcCostSquared(net, tSet, -1))
		}
		backPropagate(net, tSet, tSetNumber, alpha)
		tSetNumber = (tSetNumber + 1) % len(tSet.in)
	}
	fmt.Printf("End cost %.13f \n", calcCostSquared(net, tSet, -1))
}
