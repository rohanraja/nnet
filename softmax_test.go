package nnet

import (
	"testing"

	"github.com/rohanraja/go.matrix"
)

func TestSoftmaxGrad(t *testing.T) {

	testParams := make(Params)
	// Test NN - Input layer - 3, hidden 5, output 2

	testParams["W"] = matrix.Normals(5, 3)
	testParams["B1"] = matrix.Normals(5, 1)

	var testDataset Dataset

	testDataset.X = matrix.Normals(3, 1)
	testDataset.Y = matrix.MakeDenseMatrix([]float64{1, 0, 0, 0, 0}, 5, 1)

	sr := SoftmaxRegressor{testParams, testDataset}

	GradientCheck(sr)

}
