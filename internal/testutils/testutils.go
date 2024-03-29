package testutils

import (
	"fmt"
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

func AssertInDeltaVec(t *testing.T, expected, actual mat.Vector, rTol, aTol float64) bool {
	assert.Equal(t, expected.Len(), actual.Len())
	ok := true
	for i := 0; i < expected.Len(); i++ {
		ok = ok && InDeltaRATol(t, expected.AtVec(i), actual.AtVec(i), rTol, aTol, "Index %d", i)
	}
	return ok
}

func InDeltaRATol(t *testing.T, expected, actual, rTol, aTol float64, msgAndArgs ...any) bool {
	if aTol < 0 {
		aTol = 1e-08
	}
	if rTol < 0 {
		rTol = 1e-05
	}
	delta := (aTol + rTol*math.Abs(expected))
	if len(msgAndArgs) > 0 {
		msg := msgAndArgs[0].(string)
		msg = fmt.Sprintf("rTol: %g, aTol: %g, %s", rTol, aTol, msg)
		msgAndArgs[0] = msg
	} else {
		msgAndArgs = []any{"rTol: %g, aTol: %g", rTol, aTol}
	}
	return assert.InDelta(t, expected, actual, delta, msgAndArgs...)
}

func AssertInDeltaVecWithEndPoints(t *testing.T, expected, actual mat.Vector,
	rTolBody, rTolEnd, aTolBody, aTolEnd float64) bool {

	n := expected.Len()
	if ok := AssertInDeltaVec(
		t,
		expected.(*mat.VecDense).SliceVec(1, n-2),
		actual.(*mat.VecDense).SliceVec(1, n-2),
		rTolBody, aTolBody); !ok {
		return false
	}
	if ok := InDeltaRATol(t, expected.AtVec(0), actual.AtVec(0), rTolEnd, aTolEnd); !ok {
		return false
	}
	if ok := InDeltaRATol(t, expected.AtVec(n-1), actual.AtVec(n-1), rTolEnd, aTolEnd); !ok {
		return false
	}
	return true
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
