package main

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
	neurons []neuron
}

type net struct {
	layers []layer
	bias   float64
}
