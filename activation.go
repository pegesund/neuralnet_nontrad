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
	return val // calculate this in feed forward layer logic, keep for now
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
			setInputFirstLayer(net, tSet.in[i])
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

func calcCrossEntropy(net *net, tSet *trainingSet, maxCheck int) float64 {
	counter, sum := 0, 0.0
	lastLayer := &net.layers[len(net.layers)-1]
	for i := 0; i < len(tSet.out); i++ {
		for j := 0; j < len(lastLayer.neurons); j++ {
			setInputFirstLayer(net, tSet.in[i])
			feedForward(net)
			// seeInputOutput(*net)
			counter++
			if counter == maxCheck {
				break
			}
			sum += math.Log(lastLayer.neurons[j].out) * tSet.out[i][j]
		}
	}
	return -sum
}

func feedForwardSoftMax(net *net, i int) {
	max := net.layers[i].neurons[0].out
	for k := 1; k < len(net.layers[i].neurons); k++ {
		if net.layers[i].neurons[k].out > max {
			max = net.layers[i].neurons[k].out
		}
	}
	softMaxSum := 0.0
	for k := 0; k < len(net.layers[i].neurons); k++ {
		net.layers[i].neurons[k].out = math.Exp(net.layers[i].neurons[k].out - max)
		softMaxSum += net.layers[i].neurons[k].out
	}
	for k := 0; k < len(net.layers[i].neurons); k++ {
		newOut := net.layers[i].neurons[k].out / softMaxSum
		net.layers[i].neurons[k].out = newOut
	}
}
