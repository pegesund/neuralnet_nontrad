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

// this function is currently not used, keep for further testing
func activateSoftMaxPrime(val float64) float64 {
	return 1 // placeholder for
}

func softMaxPrimeReal(layer *layer) {

	for i := 0; i < len(layer.neurons); i++ {
		// fmt.Println("Err: ", layer.neurons[i].err, len(layer.neurons))
		layer.neurons[i].err = layer.neurons[i].err / float64(len(layer.neurons))
	}

	/*
		for i := range layer.neurons {
			fmt.Printf(", %f", layer.neurons[i].out)
		}
		fmt.Println("")
	*/

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

func correctNumberOfPredictions(net *net, tSet *trainingSet, maxCheck int) int {
	counter, sum := 0, 0
	for i := 0; i < len(tSet.out); i++ {
		setInputFirstLayer(net, tSet.in[i])
		feedForward(net)
		lastLayer := &net.layers[len(net.layers)-1]
		maxOut, maxTset := 0.0, 0.0
		maxOutCounter, maxTsetCounter := 0, 0
		for j := 0; j < len(lastLayer.neurons); j++ {
			if maxOut < lastLayer.neurons[j].out {
				maxOut = lastLayer.neurons[j].out
				maxOutCounter = j
			}
			if maxTset < lastLayer.neurons[j].out {
				maxTset = tSet.out[i][j]
				maxTsetCounter = j
			}
		}
		if maxTsetCounter == maxOutCounter {
			sum++
		}
		counter++
		if counter == maxCheck {
			break
		}

	}
	return sum
}

func calcCrossEntropy(net *net, tSet *trainingSet, maxCheck int) float64 {
	counter, sum := 0, 0.0
	for i := 0; i < len(tSet.out); i++ {
		setInputFirstLayer(net, tSet.in[i])
		feedForward(net)
		lastLayer := &net.layers[len(net.layers)-1]
		for j := 0; j < len(lastLayer.neurons); j++ {
			sum += math.Log(lastLayer.neurons[j].out) * tSet.out[i][j]
		}
		counter++
		if counter == maxCheck {
			break
		}
	}
	return -sum
}

func feedForwardSoftMax(net *net, i int) {
	max := net.layers[i].neurons[0].in
	for k := 1; k < len(net.layers[i].neurons); k++ {
		if net.layers[i].neurons[k].in > max {
			max = net.layers[i].neurons[k].in
		}
	}
	softMaxSum := 0.0
	for k := 0; k < len(net.layers[i].neurons); k++ {
		net.layers[i].neurons[k].out = math.Exp(net.layers[i].neurons[k].in - max)
		softMaxSum += net.layers[i].neurons[k].out
	}
	for k := 0; k < len(net.layers[i].neurons); k++ {
		newOut := net.layers[i].neurons[k].out / softMaxSum
		net.layers[i].neurons[k].out = newOut
	}
}
