package nnet

import (
	"testing"

	"github.com/fatih/color"
	"github.com/rohanraja/go.matrix"
)

func GetStarter() SoftmaxRegressor {

	testParams := make(Params)
	testParams["W"] = matrix.Normals(5, 3)
	testParams["B1"] = matrix.Normals(5, 1)

	var testDataset Dataset

	testDataset.X = matrix.Normals(3, 5)
	testDataset.Y = matrix.Eye(5)
	// testDataset.Y.Set(0, 0, 1)
	// testDataset.Y.Set(1, 1, 1)

	sr := SoftmaxRegressor{testParams, testDataset}

	return sr
}
func TestSGD(t *testing.T) {

	color.Magenta("Testing SGD\n")
	sr := GetStarter()
	ApplyGradToBatch(sr)

}
func TestSoftmaxGrad(t *testing.T) {

	sr := GetStarter()
	GradientCheck(sr)

}
