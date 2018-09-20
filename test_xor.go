package main

import "fmt"

/* test xor the traditional way
   one hiput layer of size 2, one hidden layer of size 2, and one output layer of size 1
   simple test with only mutations of one net
*/

func testXor() {
	layersLength := []int{2, 3, 3, 1}
	layersActivate := []ActivationFunction{Identity, Tanh, Tanh, Tanh}
	wood := createWood(3, layersLength, 0, layersActivate)
	in := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	out := [][]float64{{0}, {1}, {1}, {0}}
	tSet := trainingSet{in, out}
	training := training{&tSet, 0, 0, 0, 1000000000}
	net := cloneNet(wood.nets[0])
	net = createCloneMutateAndEvaluate(net, &training)
	predict([]float64{0, 0}, net)
	predict([]float64{1, 0}, net)
	predict([]float64{0, 1}, net)
	predict([]float64{1, 1}, net)
	fmt.Println(wood)
}

/*
	Test same input/output as last test but use softmax as the last layer
	Restructure output and training data to two neurons
	Zero is represented as {1,0} and one as {0,1}
*/
func testXorSoftMax() {
	layersLength := []int{2, 3, 3, 2}
	layersActivate := []ActivationFunction{Identity, Tanh, Tanh, SoftMax}
	wood := createWood(3, layersLength, 0, layersActivate)
	in := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	out := [][]float64{{1, 0}, {0, 1}, {0, 1}, {1, 0}}
	tSet := trainingSet{in, out}
	training := training{&tSet, 0, 0, 0, 1000000000}
	net := cloneNet(wood.nets[0])
	net = createCloneMutateAndEvaluate(net, &training)
	predict([]float64{0, 0}, net)
	predict([]float64{1, 0}, net)
	predict([]float64{0, 1}, net)
	predict([]float64{1, 1}, net)
	fmt.Println(wood)
}
