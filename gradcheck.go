package nnet

import "github.com/fatih/color"

func GradCheckGeneral(f func([]float64) float64, fprime func([]float64) []float64, x []float64) bool {

	grads := fprime(x)

	h := 0.0001
	e := 0.0001

	for i := 0; i < len(x); i++ {

		x[i] = x[i] - h
		f1 := f(x)

		x[i] = x[i] + 2*h
		f2 := f(x)

		grad := (f2 - f1) / (2 * h)
		diff := grads[i] - grad

		// color.Yellow("%f - %f = %f", grads[i], grad, diff)
		if diff > e {
			color.Red("Grad Check Failed!!\n")
			color.Yellow("%f - %f = %f", grads[i], grad, diff)
			return false
		}

	}

	color.Green("Grad Check Passed!!\n")
	return true

}

func GradCheck(params Params, dataset Dataset) bool {

	f := func(x []float64) float64 {

		var p Params
		p.Pack(x, params.GetPinfo())

		return NN_Cost(p, dataset)
	}
	fprime := func(x []float64) []float64 {

		var p Params
		p.Pack(x, params.GetPinfo())

		grads := NN_Grad(p, dataset)

		return grads.UnPack()
	}

	return GradCheckGeneral(f, fprime, params.UnPack())

}
