package main

func backPropagate(net *net, tset *trainingSet) {
	for i := 0; i < len(net.layers); i++ {
		for j := len(net.layers[i].neurons) - 2; j >= 0; j-- {
			neurone := &net.layers[i].neurons[j]
			synapses := &neurone.synapses
			for k := 0; k < len(*synapses); k++ {
			}
		}
	}
}
