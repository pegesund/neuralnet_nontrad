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
	synapses []synapse
	in       float64
	out      float64
	err      float64
}

type layer struct {
	neurons       []neuron
	activateFunc  func(float64) float64
	activateVal   ActivationFunction
	activatePrime func(float64) float64
}

type net struct {
	layers       []layer
	bias         bool
	mutationInc  float64
	layersLength []int
	error        float64
	generation   int
	netType      int
}

type wood struct {
	nets      []*net
	diversity int
}

type trainingSet struct {
	in  [][]float64
	out [][]float64
}

type darwinTraining struct {
	tSet            *trainingSet
	batchSize       int
	threads         int
	minGenerations  int
	cloneIterations int
	errPass         float64
	runGenerations  int
}

type trainMsg struct {
	training  *darwinTraining
	wood      *wood
	netNumber int
}
