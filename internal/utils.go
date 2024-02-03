package utils

import (
	"math"

	"gonum.org/v1/gonum/interp"
	"gonum.org/v1/gonum/mat"
)

// -------------------
// NON CLASS FUNCTIONS
// -------------------
func Linspace(start, stop float64, N int) *mat.VecDense {
	v := mat.NewVecDense(N, nil)
	step := (stop - start) / float64(N-1)
	for i := 0; i < N; i++ {
		v.SetVec(i, float64(i)*step)
	}
	return v
}

func ApplyVec(fn func(float64) float64, dest mat.MutableVector, src mat.Vector) mat.MutableVector {
	if dest == nil {
		dest = mat.NewVecDense(src.Len(), nil)
	}
	if dest.Len() != src.Len() {
		panic("input vectors must be the same length")
	}
	for i := 0; i < src.Len(); i++ {
		dest.SetVec(i, fn(src.AtVec(i)))
	}
	return dest
}

func Spline(x0 mat.Vector, y0 mat.Vector, x mat.Vector) mat.Vector {
	// f = interpolate.interp1d(x0, y0, axis=axis, fill_value='extrapolate', kind='cubic')
	var predictor interp.AkimaSpline
	predictor.Fit(x0.(*mat.VecDense).RawVector().Data, y0.(*mat.VecDense).RawVector().Data)
	return ApplyVec(func(x float64) (y float64) {
		var i1, i2 int
		n := x0.Len()
		switch {
		case x < x0.AtVec(0):
			i1 = 0
			i2 = 1
		case x > x0.AtVec(n-1):
			i1 = n - 1
			i2 = n - 2
		default:
			y = predictor.Predict(x)
			return
		}
		dx := x0.AtVec(i2) - x0.AtVec(i1)
		offset := x - x0.AtVec(i1)
		dy := y0.AtVec(i2) - y0.AtVec(i1)
		m := dy / dx
		if math.Abs(offset) > math.Abs(dx)*3 {
			panic("this function will only extrapolate by two segments")
		}
		y = y0.AtVec(i1) + m*offset
		return
	}, nil, x)
}
