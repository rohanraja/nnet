package nnet

import "github.com/rohanraja/go.matrix"

type SoftmaxRegressor struct {
	Parameters Params
	Data       Dataset
}

func (nn SoftmaxRegressor) Compute_Cost(params Params, datasets Dataset) float64 {

	cost := 0.0

	X := datasets.X //.GetMatrix(0, i, datasets.X.Rows(), 1)
	Y := datasets.Y //.GetMatrix(0, i, datasets.Y.Rows(), 1)

	h := matrix.Product(params["W"], X)
	h.AddVector(params["B1"])
	h.Softmax()

	delta, _ := h.ElementMultDense(Y)
	XDummy := matrix.Ones(1, delta.Rows())

	delta = matrix.Product(XDummy, delta)
	delta = delta.Log()

	XDummy = matrix.Ones(delta.Cols(), 1)

	costmat := matrix.Product(delta, XDummy)
	newcost := costmat.Array()[0]
	cost += -1 * newcost
	return cost
}

func (nn SoftmaxRegressor) Compute_Gradient(params Params, datasets Dataset) (paramGrads Params) {
	//Forward Propogation

	paramGrads = make(Params)

	X := datasets.X //.GetMatrix(0, i, datasets.X.Rows(), 1)
	Y := datasets.Y //.GetMatrix(0, i, datasets.Y.Rows(), 1)
	h := matrix.Product(params["W"], X)
	h.AddVector(params["B1"])
	h.Softmax()

	//Backward Propogation

	delta := matrix.Difference(h, Y)

	paramGrad := make(Params)
	paramGrad["W"] = matrix.Product(delta, X.Transpose())

	XDummy := matrix.Ones(delta.Cols(), 1)
	paramGrad["B1"] = matrix.Product(delta, XDummy)

	paramGrads.Add(paramGrad)

	return
}

func (nn SoftmaxRegressor) GetParams() Params {

	return nn.Parameters
}

func (nn SoftmaxRegressor) GetDataset() Dataset {

	return nn.Data
}
