package main

import (
	"fmt"
	"math"
)

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

func activateSigmoidPrime(val float64) float64 {
	return activateSigmoid(val) * (1 - activateSigmoid(val))
}

func activateTanh(val float64) float64 {
	return math.Tanh(val)
}

func activateTanhPrime(val float64) float64 {
	return 1 - math.Exp(math.Tanh(val))
}

func activateSoftMax(val float64) float64 {
	return math.Exp(val)
}

func activateSoftMaxPrime(val float64) float64 {
	return activateSoftMax(val) * (1 - activateSoftMax(val))
}

func calcLossSquared(out float64, expected float64) float64 {
	return math.Pow(expected-out, 2)
}

func calcCostSquared(net *net) float64 {
	sum := 0.0
	lastLayer := &net.layers[len(net.layers)-1]
	for i := 0; i < len(lastLayer.neurons); i++ {
		fmt.Println("Out: ", lastLayer.neurons[i].out, " - err: ", lastLayer.neurons[i].out)
		sum += lastLayer.neurons[i].err
	}
	return sum
}
