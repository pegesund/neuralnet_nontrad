package main

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"testing"
)

/* test xor the traditional way
   one hiput layer of size 2, one hidden layer of size 2, and one output layer of size 1
   simple test with only mutations of one net
*/

func TestXor(t *testing.T) {
	layersLength := []int{2, 3, 3, 1}
	layersActivate := []ActivationFunction{Identity, Tanh, Tanh, Tanh}
	wood := createWood(3, layersLength, 0.0, layersActivate)
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
func TestXorSoftMax(t *testing.T) {
	layersLength := []int{2, 3, 3, 3, 2}
	layersActivate := []ActivationFunction{Identity, Tanh, Tanh, Tanh, SoftMax}
	wood := createWood(3, layersLength, 0.2, layersActivate)
	in := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	out := [][]float64{{1, 0}, {0, 1}, {0, 1}, {1, 0}}
	//in := [][]float64{{0, 0}, {1, 1}}
	// out := [][]float64{{1, 0}, {1, 0}}
	tSet := trainingSet{in, out}
	training := darwinTraining{&tSet, 0, 0, 0, 1000000, 0.000002, 0}
	net := createCloneMutateAndEvaluate(wood.nets[0], &training)
	assert.True(t, net.error < 0.3, "Error to high in softmax")
	predict([]float64{0, 0}, net)
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
	net := initRandom(layersLength, 0.2, layersActivate)
	trainBackPropagate(net, &tSet, 0.0001, 400000, true)
	cost := calcCostSquared(net)
	fmt.Println("Cost: ", cost)
	assert.True(t, cost < 0.0001)
}

func TestBackPropTrainingXor2(t *testing.T) {
	layersLength := []int{2, 3, 3, 3, 3, 1}
	layersActivate := []ActivationFunction{Identity, Tanh, Tanh, Tanh, Tanh, Sigmoid}
	in := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	out := [][]float64{{0}, {1}, {1}, {0}}
	tSet := trainingSet{in, out}
	net := initRandom(layersLength, 0.2, layersActivate)
	trainBackPropagate(net, &tSet, 0.0001, 1000, false)
	cost := calcCostSquared(net)
	assert.True(t, cost < 0.001)
}
