package nnet

type NN_Base interface {
	Compute_Cost(Params, Dataset) float64
	Compute_Gradient(Params, Dataset) Params

	GetParams() Params
	GetDataset() Dataset
}

func GetUnpackedFunctions(nn NN_Base, dataset Dataset) (f func([]float64) float64, fprime func([]float64) []float64) {

	params := nn.GetParams()
	// dataset := nn.GetDataset()
	f = func(x []float64) float64 {

		var p Params
		p.Pack(x, params.GetPinfo())

		return nn.Compute_Cost(p, dataset)
	}
	fprime = func(x []float64) []float64 {

		var p Params
		p.Pack(x, params.GetPinfo())

		grads := nn.Compute_Gradient(p, dataset)

		return grads.UnPack()
	}

	return
}

func GradientCheck(nn NN_Base) bool {

	f, fprime := GetUnpackedFunctions(nn, nn.GetDataset())
	params := nn.GetParams()
	x := params.UnPack()

	return GradCheckGeneral(f, fprime, x)

}
