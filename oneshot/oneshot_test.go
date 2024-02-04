package oneshot_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/etfrogers/gohank"
	utils "github.com/etfrogers/gohank/internal"
	"github.com/etfrogers/gohank/internal/testutils"
	"github.com/etfrogers/gohank/oneshot"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
	"gonum.org/v1/gonum/mat"
)

const pi = math.Pi

const maxOrder int = 4

type RadialSuite struct {
	suite.Suite
	radius *mat.VecDense
}

type OneShotTestSuite struct {
	RadialSuite
	order int
}

func TestRadialSuite(t *testing.T) {
	s := new(RadialSuite)
	suite.Run(t, s)
}

func TestSuite(t *testing.T) {
	for order := 0; order <= maxOrder; order++ {
		hs := new(OneShotTestSuite)
		hs.order = order
		suite.Run(t, hs)
	}
}

func (suite *RadialSuite) SetupTest() {
	suite.radius = utils.Linspace(0, 3, 1024)
}

func kToV(k mat.Vector) mat.Vector {
	return utils.ApplyVec(func(f float64) float64 { return f / (2 * pi) }, nil, k)
}

func (t *OneShotTestSuite) TestJinc() {
	for _, a := range []float64{1, 0.7, 0.1} {
		t.Run(fmt.Sprint(a), func() {
			f := testutils.GeneralisedJinc(t.radius, a, t.order)
			kr, actual_ht := oneshot.QDHT(t.radius, f, t.order)
			v := kToV(kr)
			expected_ht := testutils.GeneralisedTopHat(v, a, t.order)
			err := testutils.MeanAbsError(expected_ht, actual_ht)
			assert.Less(t.T(), err, 1e-3)
		})
	}
}

func (t *OneShotTestSuite) TestTopHat() {
	for _, a := range []float64{1, 1.5, 0.1} {
		t.Run(fmt.Sprint(a), func() {
			f := testutils.GeneralisedTopHat(t.radius, a, t.order)
			kr, actual_ht := oneshot.QDHT(t.radius, f, t.order)
			v := kToV(kr)
			expected_ht := testutils.GeneralisedJinc(v, a, t.order)
			assert.Less(t.T(), testutils.MeanAbsError(expected_ht, actual_ht), 1e-3)
		})
	}
}

func (t *RadialSuite) TestGaussian() {
	// Note the definition in Guizar-Sicairos varies by 2*pi in
	// both scaling of the argument (so use kr rather than v) and
	// scaling of the magnitude.
	for _, a := range []float64{2, 5, 10} {
		t.Run(fmt.Sprint(a), func() {
			a2 := math.Pow(a, 2)
			f := utils.ApplyVec(func(r float64) float64 { return math.Exp(-a2 * math.Pow(r, 2)) }, nil, t.radius)

			kr, actual_ht := oneshot.QDHT(t.radius, f, 0)
			expected_ht := utils.ApplyVec(func(kr float64) float64 { return 2 * pi * (1 / (2 * a2)) * math.Exp(-math.Pow(kr, 2)/(4*a2)) },
				nil, kr)
			testutils.AssertInDeltaVec(t.T(), expected_ht, actual_ht, 0, 1e-7)
		})
	}
}

func (t *RadialSuite) TestInverseGaussian() {
	// Note the definition in Guizar-Sicairos varies by 2*pi in
	// both scaling of the argument (so use kr rather than v) and
	// scaling of the magnitude.
	for _, a := range []float64{2, 5, 10} {
		t.Run(fmt.Sprint(a), func() {
			kr := utils.Linspace(0, 200, 1024)
			a2 := math.Pow(a, 2)
			ht := utils.ApplyVec(func(kr float64) float64 { return 2 * pi * (1 / (2 * a2)) * math.Exp(-math.Pow(kr, 2)/(4*a2)) }, nil, kr)
			r, actual_f := oneshot.IQDHT(kr, ht, 0)
			expected_f := utils.ApplyVec(func(r float64) float64 { return math.Exp(-a2 * math.Pow(r, 2)) }, nil, r)
			testutils.AssertInDeltaVec(t.T(), expected_f, actual_f, 0, 1e-5)
		})
	}
}

/*
@pytest.mark.parametrize('two_d_size', [1, 35, 27])
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('a', [1, 0.7, 0.1])
@pytest.mark.parametrize('order', orders)
def test_jinc2d(radius: np.ndarray, a: float, order: int, axis: int, two_d_size: int):
    f = generalised_jinc(radius, a, order)
    second_axis = np.outer(np.linspace(0, 6, two_d_size), f)
    if axis == 0:
        f_array = np.outer(f, second_axis)
    else:
        f_array = np.outer(second_axis, f)
    kr, actual_ht = qdht(radius, f_array, axis=axis)
    v = kr / (2 * np.pi)
    expected_ht = generalised_top_hat(v, a, order)
    if axis == 0:
        expected_ht_array = np.outer(expected_ht, second_axis)
    else:
        expected_ht_array = np.outer(second_axis, expected_ht)
    error = np.mean(np.abs(expected_ht_array-actual_ht))
    # multiply tolerance to allow for the larger values caused
    # by second_axis having values greater than 1
    assert error < 1e-3 * 4

@pytest.mark.parametrize('axis', [0, 1])
def test_gaussian_2d(axis: int, radius: np.ndarray):
    # Note the definition in Guizar-Sicairos varies by 2*pi in
    # both scaling of the argument (so use kr rather than v) and
    # scaling of the magnitude.
    a = np.linspace(2, 10)
    dims_a = np.ones(2, np.int)
    dims_a[1-axis] = len(a)
    dims_r = np.ones(2, np.int)
    dims_r[axis] = len(radius)
    a_reshaped = np.reshape(a, dims_a)
    r_reshaped = np.reshape(radius, dims_r)
    f = np.exp(-a_reshaped ** 2 * r_reshaped ** 2)
    kr, actual_ht = qdht(radius, f, axis=axis)
    kr_reshaped = np.reshape(kr, dims_r)
    expected_ht = 2*np.pi*(1 / (2 * a_reshaped**2)) * np.exp(-kr_reshaped**2 / (4 * a_reshaped**2))
    assert np.allclose(expected_ht, actual_ht)


@pytest.mark.parametrize('axis', [0, 1])
def test_inverse_gaussian_2d(axis: int):
    # Note the definition in Guizar-Sicairos varies by 2*pi in
    # both scaling of the argument (so use kr rather than v) and
    # scaling of the magnitude.
    kr = np.linspace(0, 200, 1024)
    a = np.linspace(2, 10)
    dims_a = np.ones(2, np.int)
    dims_a[1-axis] = len(a)
    dims_r = np.ones(2, np.int)
    dims_r[axis] = len(kr)
    a_reshaped = np.reshape(a, dims_a)
    kr_reshaped = np.reshape(kr, dims_r)
    ht = 2*np.pi*(1 / (2 * a_reshaped**2)) * np.exp(-kr_reshaped**2 / (4 * a_reshaped**2))
    r, actual_f = iqdht(kr, ht, axis=axis)
    r_reshaped = np.reshape(r, dims_r)
    expected_f = np.exp(-a_reshaped ** 2 * r_reshaped ** 2)
    assert np.allclose(expected_f, actual_f)
*/

/*
//////// Below test not implmented as it requires a Bessel K function
@pytest.mark.parametrize('a', [2, 1, 0.5])
def test_1_over_r2_plus_z2(a: float):
    # Note the definition in Guizar-Sicairos varies by 2*pi in
    # both scaling of the argument (so use kr rather than v) and
    # scaling of the magnitude.
    r = np.linspace(0, 50, 1024)
    f = 1 / (r**2 + a**2)
    # kn cannot handle complex arguments, so a must be real
    kr, actual_ht = qdht(r, f)
    expected_ht = 2 * np.pi * scipy_bessel.kn(0, a * kr)
    # as this diverges at zero, the first few entries have higher errors, so ignore them
    expected_ht = expected_ht[10:]
    actual_ht = actual_ht[10:]
    error = np.mean(np.abs(expected_ht - actual_ht))
    assert error < 1e-3
*/

// -------------------
// Test equivalence of one-shot and standard
// -------------------

func (t *OneShotTestSuite) TestJincEquivalence() {
	for _, a := range []float64{1, 0.7, 0.1} {
		t.Run(fmt.Sprint(a), func() {
			transformer := gohank.NewTransformFromRadius(t.order, t.radius)
			f := testutils.GeneralisedJinc(t.radius, a, t.order)
			_, one_shot_ht := oneshot.QDHT(t.radius, f, t.order)

			f_t := testutils.GeneralisedJinc(transformer.Radius(), a, transformer.Order())
			standard_ht := transformer.QDHT(f_t)
			testutils.AssertInDeltaVec(t.T(), one_shot_ht, standard_ht, 0, 1e-6)
		})
	}
}

func TestTopHatEquivalence(t *testing.T) {
	t.Skip("generalised_top_hat has discontinuities, so deals badly with interpolation")
	// transformer = HankelTransform(order=order, radial_grid=radius)
	// f = generalised_top_hat(radius, a, order)
	// kr, one_shot_ht = qdht(radius, f, order=order)

	// f_t = generalised_top_hat(transformer.r, a, transformer.order)
	// standard_ht = transformer.qdht(f_t)
	// assert np.allclose(one_shot_ht, standard_ht)
}

func (t *RadialSuite) TestGaussianEquivalence() {
	// Note the definition in Guizar-Sicairos varies by 2*pi in
	// both scaling of the argument (so use kr rather than v) and
	// scaling of the magnitude.
	for _, a := range []float64{2, 5, 10} {
		t.Run(fmt.Sprint(a), func() {
			transformer := gohank.NewTransformFromRadius(0, t.radius)
			a2 := math.Pow(a, 2)
			f := utils.ApplyVec(func(r float64) float64 { return math.Exp(-a2 * math.Pow(r, 2)) }, nil, t.radius)
			_, one_shot_ht := oneshot.QDHT(t.radius, f, 0)

			f_t := utils.ApplyVec(func(r float64) float64 { return math.Exp(-a2 * math.Pow(r, 2)) }, nil, transformer.Radius())
			standard_ht := transformer.QDHT(f_t)
			testutils.AssertInDeltaVec(t.T(), one_shot_ht, standard_ht, 1e-3, 1e-4)
		})
	}
}

func Test1OverR2PlusZ2Qquivalence(t *testing.T) {
	for _, a := range []float64{2, 1, 4} {
		t.Run(fmt.Sprint(a), func(t *testing.T) {
			a2 := math.Pow(a, 2)
			shape := func(r float64) float64 { return 1 / (math.Pow(r, 2) + a2) }

			r := utils.Linspace(0, 50, 1024)
			f := utils.ApplyVec(shape, nil, r)
			transformer := gohank.NewTransformFromRadius(0, r)
			f_transformer := utils.ApplyVec(shape, nil, r)

			testutils.AssertInDeltaVecWithEndPoints(t, transformer.ToTransformR(f), f_transformer, 2e-2, 1e-2, 2e-2, 1e-2)

			kr, one_shot_ht := oneshot.QDHT(r, f, 0)
			testutils.AssertInDeltaVecWithEndPoints(t, kr, transformer.Kr(), 1e-2, -1, 1e-6, -1)
			standard_ht := transformer.QDHT(f_transformer)

			testutils.AssertInDeltaVecWithEndPoints(t, standard_ht, one_shot_ht, 2e-2, 2e-2, 1e-1, 1e-2)
		})
	}
}
