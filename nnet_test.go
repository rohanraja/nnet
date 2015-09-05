package nnet

import (
	"testing"

	"github.com/fatih/color"
	"github.com/rohanraja/go.matrix"
)

func TestNNGrad(t *testing.T) {

	// var testParams Params
	testParams := make(Params)
	// Test NN - Input layer - 3, hidden 5, output 2

	testParams["W"] = matrix.Normals(5, 3)
	testParams["B1"] = matrix.Normals(5, 1)

	var testDataset Dataset

	testDataset.X = matrix.Normals(3, 1)
	testDataset.Y = matrix.MakeDenseMatrix([]float64{1, 0, 0, 0, 0}, 5, 1)

	// color.Cyan("%v", testParams)
	// color.Cyan("%+v", testDataset)

	cost := NN_Cost(testParams, testDataset)
	color.Yellow("\nCost = %f", cost)
	NN_Grad(testParams, testDataset)
}
