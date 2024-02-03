package testutils

import (
	"math"
	"testing"

	utils "github.com/etfrogers/gohank/internal"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

// ----------------
// HELPER FUNCTIONS
// ----------------
func MeanAbsError(v1, v2 mat.Vector) float64 {
	if v1.Len() != v2.Len() {
		panic("vector sizes mismatched")
	}
	length := v1.Len()
	sum := 0.
	for i := 0; i < length; i++ {
		err := math.Abs(v1.AtVec(i) - v2.AtVec(i))
		sum += err
	}
	return sum / float64(length)
}

const ATOL_LIMIT = 2e-5

func AssertInDeltaVec(t *testing.T, expected, actual mat.Vector, precision float64, relativeTol bool) {
	assert.Equal(t, expected.Len(), actual.Len())
	for i := 0; i < expected.Len(); i++ {
		tol := precision
		exp := expected.AtVec(i)
		if relativeTol {
			tol = math.Abs(precision * exp)
			if tol < ATOL_LIMIT {
				tol = ATOL_LIMIT
			}
		}
		InDeltaRTol(t, exp, actual.AtVec(i), tol, relativeTol, "Index %d", i)
	}
}

func InDeltaRTol(t *testing.T, expected, actual, precision float64, relativeTol bool, msgAndArgs ...any) {
	tol := precision
	if relativeTol {
		tol = math.Abs(precision * expected)
		if tol < ATOL_LIMIT {
			tol = ATOL_LIMIT
		}
	}
	assert.InDelta(t, expected, actual, tol, msgAndArgs...)
}

func AssertInDeltaVecWithEndPoints(t *testing.T, expected, actual mat.Vector,
	precisionBody, precisionEnd float64, relativeTol bool) {

	n := expected.Len()
	AssertInDeltaVec(
		t,
		expected.(*mat.VecDense).SliceVec(1, n-2),
		actual.(*mat.VecDense).SliceVec(1, n-2),
		precisionBody, relativeTol)
	InDeltaRTol(t, expected.AtVec(0), actual.AtVec(0), precisionEnd, relativeTol)
	InDeltaRTol(t, expected.AtVec(n-1), actual.AtVec(n-1), precisionEnd, relativeTol)
}

// ---------------
// MATHS FUNCTIONS
// ----------------
func GeneralisedTopHat(r mat.Vector, a float64, p int) mat.Vector {
	f := utils.ApplyVec(func(val float64) float64 { return generalisedTopHatF(val, a, p) }, nil, r)
	return f
}

func generalisedTopHatF(r float64, a float64, p int) float64 {
	var val float64
	if r <= a {
		val = math.Pow(r, float64(p))
	}
	// othwerise 0

	return val
}

func GeneralisedJinc(v mat.Vector, a float64, p int) mat.Vector {
	f := utils.ApplyVec(func(val float64) float64 { return generalisedJincF(val, a, p) }, nil, v)
	return f
}

func generalisedJincF(v float64, a float64, p int) float64 {

	var val float64
	if v == 0. {
		switch {
		case p == -1:
			val = math.Inf(1)
		case p == -2:
			val = -math.Pi
		case p == 0:
			val = math.Pi * math.Pow(a, 2)
		default:
			val = 0
		}
	} else {
		prefactor := math.Pow(a, float64(p+1))
		x := 2 * math.Pi * a * v
		j := math.Jn(p+1, x)
		val = prefactor * j / v
	}

	return val
}
