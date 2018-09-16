package main

import "fmt"

// generate xor for all numbers from 255
// input in the binary 1 or one for each byte

// generates bit values
func testXor() {
	in := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	out := [][]float64{{0}, {1}, {1}, {0}}
	traningSet := trainingSet{in, out}
	fmt.Println("trainingset", traningSet)
}
