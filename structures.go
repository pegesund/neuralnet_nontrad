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
	layers         []layer
	bias           float64
	mutationInc    float64
	layersLength   []int
	layersActivate []func(float64) float64
	layersActVal   []ActivationFunction
	error          float64
}

type wood struct {
	nets      []*net
	diversity int
}

type trainingSet struct {
	in  [][]float64
	out [][]float64
}

type training struct {
	tSet      *trainingSet
	rndSize   int
	threads   int
	errSize   float64
	swapInter int
}
