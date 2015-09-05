package nnet

import "github.com/rohanraja/go.matrix"

type SoftmaxRegressor struct {
	Parameters Params
	Data       Dataset
}

func (nn SoftmaxRegressor) Compute_Cost(params Params, dataset Dataset) float64 {

	h := matrix.Product(params["W"], dataset.X)
	h.Add(params["B1"])

	h.Softmax()

	delta := matrix.Product(h.Transpose(), dataset.Y)

	costMat := delta.Log()
	cost := costMat.Array()[0]
	return cost
}

func (nn SoftmaxRegressor) Compute_Gradient(params Params, dataset Dataset) (paramGrads Params) {
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

func (nn SoftmaxRegressor) GetParams() Params {

	return nn.Parameters
}

func (nn SoftmaxRegressor) GetDataset() Dataset {

	return nn.Data
}
