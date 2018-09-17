package main

/*
type training struct {
	tSet         *[]trainingSet
	rndSize		 int
	threads      int
	errSize      float64
	swapInter	 int
}
 */

// text xor
// one hiput layer of size 2, one hidden layer of size 2, and one output layer of size 1
func testXor() (*training){
	in := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	out := [][]float64{{0}, {1}, {1}, {0}}
	tSet := trainingSet{in, out}
	training := training{&tSet, 0, 0, 0, 1000000000}
	return &training
}
