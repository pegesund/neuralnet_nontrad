package main

/*
	sets the error value in neurones in last layer
*/
func updateErrorInLastLayer(net *net, tSet *trainingSet, tSetNumber int) {
	lastLayer := &net.layers[len(net.layers)-1]
	for i := 0; i < len(tSet.out); i++ {
		lastLayer.neurons[i].err = calcLossMeanSquared(lastLayer.neurons[i].out, tSet.out[tSetNumber][i])
	}
}

func backPropagate(net *net, tSet *trainingSet, tSetNumber int) {
	updateErrorInLastLayer(net, tSet, tSetNumber)

	// calc layer backwords from next layer
	for i := len(net.layers) - 2; i >= 0; i++ {
		for j := 0; j < len(net.layers[i].neurons); j++ {
			neurone := &net.layers[i].neurons[j]
			synapses := &neurone.synapses
			neurone.err = 0.0
			for k := 0; k < len(*synapses); k++ {
				synapse := &net.layers[i].neurons[k].synapses[k]
				toNeurone := &net.layers[i+1].neurons[k]
				neurone.err += toNeurone.err * (synapse.weight / toNeurone.in)
			}

		}
	}
}
