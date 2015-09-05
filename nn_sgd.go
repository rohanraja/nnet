package nnet

import "github.com/fatih/color"

func ApplyGradToBatch(nn NN_Base) {

	alpha := 1.3

	params := nn.GetParams()
	theta := params.UnPack()

	for i := 0; i < 100; i++ {

		f, fprime := GetUnpackedFunctions(nn, nn.GetDataset())
		cost := f(theta)
		color.Green("Cost = %f", cost)

		grads := fprime(theta)

		for i := 0; i < len(theta); i++ {

			theta[i] = theta[i] - alpha*grads[i]
		}

	}

}
