package main

const (
	Increase   float64 = 1
	Decrease   float64 = -1
	ClonedNet  int     = 1
	MergedNet  int     = 2
	NewRandNet int     = 3
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
	generation     int
	netType        int
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
	tSet            *trainingSet
	batchSize       int
	threads         int
	minGenerations  int
	cloneIterations int
	errPass         float64
	runGenerations  int
}

type trainMsg struct {
	training  *training
	wood      *wood
	netNumber int
}
