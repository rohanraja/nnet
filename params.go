package nnet

import (
	"sort"

	"github.com/rohanraja/go.matrix"
)

type Params map[string]*matrix.DenseMatrix

type Dimention struct {
	Rows int
	Cols int
}

type Paraminfo map[string]Dimention

func (p1 *Params) Add(p2 Params) {

	for key, _ := range p2 {

		_, ok := (*p1)[key]

		if ok == false {

			(*p1)[key] = matrix.Zeros(p2[key].Rows(), p2[key].Cols())
		}

		(*p1)[key].Add(p2[key])
	}

}

func (p *Params) Pack(inp []float64, pinfo Paraminfo) {

	pnew := make(Params)
	st := 0
	end := 0
	var keys []string
	for k, _ := range pinfo {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, key := range keys {

		val := pinfo[key]
		end = end + val.Rows*val.Cols
		mat := inp[st:end]
		st = st + end
		pnew[key] = matrix.MakeDenseMatrix(mat, val.Rows, val.Cols)
	}

	*p = pnew

}

func (p *Params) GetPinfo() (pinfo Paraminfo) {

	pinfo = make(Paraminfo)
	var keys []string
	for k, _ := range *p {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, key := range keys {

		val := (*p)[key]

		pinfo[key] = Dimention{val.Rows(), val.Cols()}
	}
	return
}

func (p *Params) UnPack() (out []float64) {
	var keys []string
	for k, _ := range *p {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, key := range keys {

		val := (*p)[key]

		out = append(out, val.Array()...)
	}
	return
}
