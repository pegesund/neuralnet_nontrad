package main

import "math"

type ActivationFunction int

const (
	Identity ActivationFunction = 0
	Sigmoid  ActivationFunction = 1
	Tanh     ActivationFunction = 2
	SoftMax  ActivationFunction = 3
)

func activateIdentity(val float64) float64 {
	return val
}

func activateSigmoid(val float64) float64 {
	return 1 / (1 + math.Exp(-val))
}

func activateTanh(val float64) float64 {
	return math.Tanh(val)
}

func activateSoftMax(val float64) float64 {
	return math.Exp(val)
}