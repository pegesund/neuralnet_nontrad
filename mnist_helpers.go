package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func createMnistDataset(fileName string) trainingSet {
	file, err := os.Open(fileName)
	if err != nil {
		panic(err)
	}
	in := make([][]float64, 0)
	out := make([][]float64, 0)
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		nums := strings.Split(line, ",")
		lineIn := make([]float64, len(nums)-1)
		lineOut := make([]float64, 10)
		val, _ := strconv.Atoi(nums[0])
		lineOut[val] = 1.0
		for i := 1; i < len(nums); i++ {
			val, _ := strconv.Atoi(nums[i])
			lineIn[i-1] = float64(val) / 255.0
		}
		in = append(in, lineIn)
		out = append(out, lineOut)
		// fmt.Println(len(lineIn))
	}
	fmt.Println("Dataset is read")
	return trainingSet{in, out}
}
