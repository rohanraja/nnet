package nnet

import "github.com/rohanraja/go.matrix"

type Dataset struct {
	X *matrix.DenseMatrix
	Y *matrix.DenseMatrix
}

func NN_Cost(params Params, dataset Dataset) float64 {

	h := matrix.Product(params["W"], dataset.X)
	h.Add(params["B1"])

	h.Softmax()

	delta := matrix.Product(h.Transpose(), dataset.Y)

	costMat := delta.Log()
	cost := costMat.Array()[0]
	return cost
}

func NN_Grad(params Params, dataset Dataset) (paramGrads Params) {
	//Forward Propogation

	h := matrix.Product(params["W"], dataset.X)
	h.Add(params["B1"])
	h.Softmax()

	//Backward Propogation

	delta := matrix.Difference(dataset.Y, h)

	paramGrads = make(Params)
	paramGrads["W"] = matrix.Product(delta, dataset.X.Transpose())
	paramGrads["B1"] = delta

	// color.Cyan("%v", paramGrads)
	return
}
