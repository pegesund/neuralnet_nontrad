package main

import (
	"fmt"
)

type Direction int
const (
	Increase    Direction = 0
	Decrease    Direction = 1
	Stay		Direction = 2
)

type neuron struct {
	weight float64
	direction Direction
	incSize float64
}

type layer struct {
	row []neuron
}

type net struct {
	rows []layer
}

func init_random(layers []int) {
	var net = net{make([]layer, len(layers)) }
	for i := 0; i < len(layers); i++ {
		net.rows[i] = layer{make([]neuron, layers[i]) }
	}
	fmt.Println(net)

}

func main() {
	layers := [2]int{3, 2}

	// f := rand.Float64()

	// fmt.Println(layers, f)
	init_random(layers[:])
}
