package main

import (
	"fmt"
	"github.com/kr/pretty"
	"math/rand"
)

type Direction int

const (
	Increase Direction = 0
	Decrease Direction = 1
	Stay     Direction = 2
)

// data structures
// structure is like a fully connected neural network

type synapse struct {
	weight    float64
	direction Direction
	incSize   float64
}

type neuron struct {
	val      float64
	synapses []synapse
}

type layer struct {
	layer []neuron
}

type net struct {
	net []layer
}

func see_net(net net) {
	pretty.Println(net)
}

func get_random_val() float64 {
	return rand.Float64()
}

func init_random(layers []int) {
	var net = net{make([]layer, len(layers))}
	for i := 0; i < len(layers); i++ {
		layerLen := layers[i]
		layer := layer{make([]neuron, layerLen)}
		for j, _ := range layer.layer {
			layer.layer[j].val = 0
			if i < layerLen-2 {
				synapses := make([]synapse, layers[i+1])
				for k, _ := range synapses {
					synapses[k].weight = get_random_val()
					synapses[k].incSize = get_random_val() / 10
					if rand.Intn(2) == 1 {
						synapses[k].direction = 1
					} else {
						synapses[k].direction = -1
					}

				}
				layer.layer[j].synapses = synapses
			} else {
				layer.layer[j].synapses = make([]synapse, 0)
			}
		}
		net.net[i] = layer
	}
	fmt.Println(net)
	see_net(net)
}

func main() {
	layers := []int{3, 3}

	// f := rand.Float64()

	// fmt.Println(layers, f)
	init_random(layers[:])
}
