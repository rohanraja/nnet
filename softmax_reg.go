package nnet

import "github.com/rohanraja/go.matrix"

type SoftmaxRegressor struct {
	Parameters Params
	Data       Dataset
}

func (nn SoftmaxRegressor) Compute_Cost(params Params, datasets Dataset) float64 {

	cost := 0.0
	for i := 0; i < datasets.X.Cols(); i++ {

		X := datasets.X.GetMatrix(0, i, datasets.X.Rows(), 1)
		Y := datasets.Y.GetMatrix(0, i, datasets.Y.Rows(), 1)

		h := matrix.Product(params["W"], X)
		h.Add(params["B1"])

		h.Softmax()

		delta := matrix.Product(h.Transpose(), Y)

		costMat := delta.Log()
		newcost := costMat.Array()[0]
		cost += -1 * newcost
	}
	return cost
}

func (nn SoftmaxRegressor) Compute_Gradient(params Params, datasets Dataset) (paramGrads Params) {
	//Forward Propogation

	paramGrads = make(Params)
	for i := 0; i < datasets.X.Cols(); i++ {
		X := datasets.X.GetMatrix(0, i, datasets.X.Rows(), 1)
		Y := datasets.Y.GetMatrix(0, i, datasets.Y.Rows(), 1)
		h := matrix.Product(params["W"], X)
		h.Add(params["B1"])
		h.Softmax()

		//Backward Propogation

		delta := matrix.Difference(h, Y)

		paramGrad := make(Params)
		paramGrad["W"] = matrix.Product(delta, X.Transpose())
		paramGrad["B1"] = delta

		paramGrads.Add(paramGrad)

	}
	// color.Cyan("%v", paramGrads)
	return
}

func (nn SoftmaxRegressor) GetParams() Params {

	return nn.Parameters
}

func (nn SoftmaxRegressor) GetDataset() Dataset {

	return nn.Data
}
