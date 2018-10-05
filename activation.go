package main

import (
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
	return val * (1 - val)
}

func activateTanh(val float64) float64 {
	return math.Tanh(val)
}

func activateTanhPrime(val float64) float64 {
	return 1 - math.Pow(val, 2)
}

func activateSoftMax(val float64) float64 {
	return math.Exp(val)
}

func activateSoftMaxPrime(val float64) float64 {
	return val * (1 - val)
}

func calcLossSquared(expected float64, out float64) float64 {
	return math.Pow(expected-out, 2)
}

/*
	Calc loss squared against a training set, start from beginning
    maxCheck iterations, -1 for all
*/
func calcCostSquared(net *net, tSet *trainingSet, maxCheck int) float64 {
	counter, sum := 0, 0.0
	lastLayer := &net.layers[len(net.layers)-1]
	for i := 0; i < len(tSet.out); i++ {
		for j := 0; j < len(lastLayer.neurons); j++ {
			setInputFirstLayer(net, tSet.out[i])
			feedForward(net)
			counter++
			if counter == maxCheck {
				break
			}
			sum += calcLossSquared(tSet.out[i][j], lastLayer.neurons[j].out)
		}
	}
	return sum / float64(counter)
}
