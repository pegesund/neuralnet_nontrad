package main

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"math/rand"
	"testing"
)

/* test xor the traditional way
   one hiput layer of size 2, one hidden layer of size 2, and one output layer of size 1
   simple test with only mutations of one net
*/

func TestXorDarwin(t *testing.T) {
	layersLength := []int{2, 3, 3, 1}
	layersActivate := []ActivationFunction{Identity, Tanh, Tanh, Tanh}
	wood := createWood(3, layersLength, false, layersActivate)
	in := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	out := [][]float64{{0}, {1}, {1}, {0}}
	tSet := trainingSet{in, out}
	training := darwinTraining{&tSet, 0, 3, 7, 10000000, 0.00005, 0}
	net := createCloneMutateAndEvaluate(wood.nets[0], &training)
	assert.True(t, net.error < 0.01, "Error to high in tanh")
	predict([]float64{0, 0}, net)
	assert.True(t, net.layers[len(net.layersLength)-1].neurons[0].out < 0.1)
	predict([]float64{1, 0}, net)
	assert.True(t, net.layers[len(net.layersLength)-1].neurons[0].out > 0.9)
	predict([]float64{0, 1}, net)
	assert.True(t, net.layers[len(net.layersLength)-1].neurons[0].out > 0.9)
	predict([]float64{1, 1}, net)
	assert.True(t, net.layers[len(net.layersLength)-1].neurons[0].out < 0.1)
	fmt.Println(wood)
}

/*
	Test same input/output as last test but use softmax as the last layer
	Restructure output and darwinTraining data to two neurons
	Zero is represented as {1,0} and one as {0,1}
*/
func TestXorSoftMaxDarwin(t *testing.T) {
	layersLength := []int{2, 3, 3, 3, 2}
	layersActivate := []ActivationFunction{Identity, Tanh, Tanh, Tanh, SoftMax}
	wood := createWood(3, layersLength, true, layersActivate)
	in := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	out := [][]float64{{1, 0}, {0, 1}, {0, 1}, {1, 0}}
	//in := [][]float64{{0, 0}, {1, 1}}
	// out := [][]float64{{1, 0}, {1, 0}}
	tSet := trainingSet{in, out}
	training := darwinTraining{&tSet, 0, 0, 0, 1000000, 0.000002, 0}
	net := createCloneMutateAndEvaluate(wood.nets[0], &training)
	assert.True(t, net.error < 0.3, "Error to high in softmax")
	predict([]float64{0, 0}, net)
	seeInputOutput(*net)
	assert.True(t, net.layers[len(net.layersLength)-1].neurons[0].out > 0.99)
	assert.True(t, net.layers[len(net.layersLength)-1].neurons[1].out < 0.01)
	predict([]float64{1, 0}, net)
	assert.True(t, net.layers[len(net.layersLength)-1].neurons[1].out > 0.99)
	assert.True(t, net.layers[len(net.layersLength)-1].neurons[0].out < 0.01)
	predict([]float64{0, 1}, net)
	assert.True(t, net.layers[len(net.layersLength)-1].neurons[1].out > 0.99)
	assert.True(t, net.layers[len(net.layersLength)-1].neurons[0].out < 0.01)
	predict([]float64{1, 1}, net)
	assert.True(t, net.layers[len(net.layersLength)-1].neurons[0].out > 0.99)
	assert.True(t, net.layers[len(net.layersLength)-1].neurons[1].out < 0.01)
	fmt.Println(wood)
}

func TestBackPropTrainingXor1(t *testing.T) {
	layersLength := []int{2, 3, 3, 3, 3, 1}
	layersActivate := []ActivationFunction{Identity, Tanh, Tanh, Tanh, Tanh, Tanh}
	in := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	out := [][]float64{{0}, {1}, {1}, {0}}
	tSet := trainingSet{in, out}
	net := initRandom(layersLength, true, layersActivate)
	trainBackPropagate(net, &tSet, nil, 0.4, 1000001, 0, false)
	cost := calcCostSquared(net, &tSet, -1)
	assert.True(t, cost < 0.0001)
}

func TestBackPropTrainingXor2(t *testing.T) {
	rand.Seed(0)
	layersLength := []int{2, 3, 3, 3, 3, 1}
	layersActivate := []ActivationFunction{Identity, Sigmoid, Sigmoid, Sigmoid, Sigmoid, Sigmoid}
	in := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	out := [][]float64{{0}, {1}, {1}, {0}}
	tSet := trainingSet{in, out}
	net := initRandom(layersLength, true, layersActivate)
	trainBackPropagate(net, &tSet, nil, 0.6, 10000000, 0.4, false)
	cost := calcCostSquared(net, &tSet, -1)
	assert.True(t, cost < 0.01)
}

func TestXorBackpropSoftmax(t *testing.T) {
	layersLength := []int{2, 3, 2}
	layersActivate := []ActivationFunction{Identity, Tanh, SoftMax}
	in := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	out := [][]float64{{1, 0}, {0, 1}, {0, 1}, {1, 0}}
	tSet := trainingSet{in, out}
	net := initRandom(layersLength, true, layersActivate)
	trainBackPropagate(net, &tSet, nil, 0.4, 1000000, 0, false)
	seeInputOutput(*net)
	cost := calcCrossEntropy(net, &tSet, -1)
	assert.True(t, cost < 0.01)
}
