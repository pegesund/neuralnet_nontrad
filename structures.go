package main


const (
	Increase float64 = 1
	Decrease float64 = -1
	Stay     float64 = 2
)

// data structures
// structure is like a fully connected neural network

type synapse struct {
	weight    float64
	direction float64
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
	mutationInc float64
}

type trainingset struct {
	in []float64
	out []float64
}

type training struct {
	net *net
	tSet *trainingset
	traningSize int
	threads int
	bucketSize int
	errSize float64
}