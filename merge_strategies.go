package main

import "math/rand"

func mergeNetsRandom(mother *net, father *net) *net {
	child := cloneNet(mother)
	treeSize := len(mother.layers)
	for i := 0; i < treeSize; i++ {
		for j := 0; j < len(mother.layers[i].neurons); j++ {
			inheritFather := rand.Intn(2) == 1
			if inheritFather {
				child.layers[i].neurons[j].val = father.layers[i].neurons[j].val
				child.bias = father.bias
			}
			for k := 0; k < len(mother.layers[i].neurons[j].synapses); k++ {
				inheritFather := rand.Intn(2) == 1
				if inheritFather {
					child.layers[i].neurons[j].synapses[k].direction = father.layers[i].neurons[j].synapses[k].direction
					child.layers[i].neurons[j].synapses[k].weight = father.layers[i].neurons[j].synapses[k].weight
					child.layers[i].neurons[j].synapses[k].incSize = father.layers[i].neurons[j].synapses[k].incSize
				}
			}
		}
	}
	child.generation = 0
	child.netType = MergedNet
	return child
}
