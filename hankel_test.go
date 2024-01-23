package gohank

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

/*from typing import Callable

import numpy as np
import pytest
import scipy.special as scipy_bessel

from pyhank import HankelTransform


smooth_shapes = [lambda r: np.exp(-r ** 2),
                 lambda r: r,
                 lambda r: r ** 2,
                 lambda r: 1 / np.sqrt(r**2 + 0.1**2)]

all_shapes = smooth_shapes.copy()
all_shapes.append(lambda r: np.random.random(r.size))

orders = list(range(0, 5))

@pytest.fixture(params=orders)
def transformer(request, radius) -> HankelTransform:
    order = request.param
    return HankelTransform(order, radial_grid=radius)


@pytest.mark.parametrize('shape', all_shapes)
def test_parsevals_theorem(shape: Callable,
                           radius: np.ndarray,
                           transformer: HankelTransform):
    # As per equation 11 of Guizar-Sicairos, the UNSCALED transform is unitary,
    # i.e. if we pass in the unscaled fr (=Fr), the unscaled fv (=Fv)should have the
    # same sum of abs val^2. Here the unscale transform is simply given by
    # ht = transformer.T @ func
    func = shape(radius)
    intensity_before = np.abs(func)**2
    energy_before = np.sum(intensity_before)
    ht = transformer.T @ func
    intensity_after = np.abs(ht)**2
    energy_after = np.sum(intensity_after)
    assert np.isclose(energy_before, energy_after)


@pytest.mark.parametrize('shape', [generalised_jinc, generalised_top_hat])
def test_energy_conservation(shape: Callable,
                             transformer: HankelTransform):
    transformer = HankelTransform(transformer.order, 10, transformer.n_points)
    func = shape(transformer.r, 0.5, transformer.order)
    intensity_before = np.abs(func)**2
    energy_before = np.trapz(y=intensity_before * 2 * np.pi * transformer.r,
                             x=transformer.r)

    ht = transformer.qdht(func)
    intensity_after = np.abs(ht)**2
    energy_after = np.trapz(y=intensity_after * 2 * np.pi * transformer.v,
                            x=transformer.v)
    assert np.isclose(energy_before, energy_after, rtol=0.01)

*/

type HankelTestSuite struct {
	suite.Suite
	radius      mat.VecDense
	transformer HankelTransform
	order       int
}

func (suite *HankelTestSuite) SetupTest() {
	suite.radius = *linspace(0, 3, 1024)
	suite.transformer = NewTransformFromRadius(suite.order, &suite.radius)
}

func randomVecLike(shape mat.Vector) mat.Vector {
	n, _ := shape.Dims()
	return mat.NewVecDense(n, nil)
}

func (t *HankelTestSuite) TestRoundTrip() {
	fun := randomVecLike(&t.radius)
	ht := t.transformer.QDHT(fun)
	reconstructed := t.transformer.IQDHT(ht)
	assertInDeltaVec(t.T(), fun, reconstructed, 1e-9)
}

func (t *HankelTestSuite) TestTopHat() {
	for _, a := range []float64{1, 1.5, 0.1} {
		t.Run(fmt.Sprint(a), func() {
			f := generalisedTopHat(&t.transformer.r, a, t.order)
			plotVec("Top Hat", &t.transformer.r, f)
			expected_ht := generalisedJinc(&t.transformer.v, a, t.order)
			actual_ht := t.transformer.QDHT(f)
			plotVec("Jinc", &t.transformer.v, expected_ht, actual_ht)
			assert.Less(t.T(), meanAbsError(expected_ht, actual_ht), 1e-3)
		})

	}
}

func TestGeneralisedJincZero(t *testing.T) {
	for _, a := range []float64{1, 0.7, 0.1, 136., 1e-6} {
		for p := -10; p < 10; p++ {
			t.Run(fmt.Sprintf("%f, %d", a, p), func(t *testing.T) {
				if p == -1 {
					t.Skip("Skipping test for p=-1 as 1/eps does not go to inf correctly")
				}
				eps := 1e-200
				if p == -2 {
					eps = 1e-5 / a
				}
				v := mat.NewVecDense(2, []float64{0, eps})
				val := generalisedJinc(v, a, p)

				tolerance := 2e-9
				assert.InDelta(t, val.AtVec(0), val.AtVec(1), tolerance)
			})
		}
	}
}

func meanAbsError(v1, v2 mat.Vector) float64 {
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

func assertInDeltaVec(t *testing.T, expected, actual mat.Vector, precision float64) {
	assert.Equal(t, expected.Len(), actual.Len())
	for i := 0; i < expected.Len(); i++ {
		assert.InDelta(t, expected.AtVec(i), actual.AtVec(i), precision)
	}
}

func TestSuite(t *testing.T) {
	maxOrder := 4
	for order := 0; order <= maxOrder; order++ {
		hs := new(HankelTestSuite)
		hs.order = order
		suite.Run(t, hs)
	}
}

func generalisedTopHat(r mat.Vector, a float64, p int) *mat.VecDense {
	topHat := mat.NewVecDense(r.Len(), nil)
	for i := 0; i < r.Len(); i++ {
		if r.AtVec(i) <= a {
			topHat.SetVec(i, math.Pow(r.AtVec(i), float64(p)))
		}
		// othwerise 0
	}
	return topHat
}

func generalisedJinc(v mat.Vector, a float64, p int) *mat.VecDense {
	val := mat.NewVecDense(v.Len(), nil)
	for i := 0; i < v.Len(); i++ {
		v_ := v.AtVec(i)
		if v_ == 0. {
			switch {
			case p == -1:
				val.SetVec(i, math.Inf(1))
			case p == -2:
				val.SetVec(i, -math.Pi)
			case p == 0:
				val.SetVec(i, math.Pi*math.Pow(a, 2))
			default:
				val.SetVec(i, 0)
			}
		} else {
			prefactor := math.Pow(a, float64(p+1))
			x := 2 * math.Pi * a * v.AtVec(i)
			j := math.Jn(p+1, x)
			elem := prefactor * j / v.AtVec(i)
			val.SetVec(i, elem)
		}
	}
	return val
}

func plotVec(name string, x mat.Vector, ys ...mat.Vector) {
	p := plot.New()
	p.Title.Text = name
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	for i, y := range ys {
		lpLine, _, err := plotter.NewLinePoints(pointsFromVec(x, y))
		lpLine.LineStyle.Color = plotutil.DefaultColors[i]
		// err := plotutil.AddLinePoints(p,
		// 	fmt.Sprint(i), pointsFromVec(x, y),
		// 	// "Second", randomPoints(15),
		// 	// "Third", randomPoints(15)
		// )
		p.Add(lpLine)
		if err != nil {
			panic(err)
		}
	}

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, name+".png"); err != nil {
		panic(err)
	}
}

func pointsFromVec(x, y mat.Vector) plotter.XYs {
	pts := make(plotter.XYs, x.Len())
	for i := range pts {
		pts[i].X = x.AtVec(i)
		pts[i].Y = y.AtVec(i)
	}
	return pts
}

/*

@pytest.mark.parametrize('two_d_size', [1, 100, 27])
@pytest.mark.parametrize('axis', [0, 1])
def test_round_trip_2d(two_d_size: int, axis: int, radius: np.ndarray, transformer: HankelTransform):
    dims = np.ones(2, np.int) * two_d_size
    dims[axis] = radius.size
    func = np.random.random(dims)
    ht = transformer.qdht(func, axis=axis)
    reconstructed = transformer.iqdht(ht, axis=axis)
    assert np.allclose(func, reconstructed)


@pytest.mark.parametrize('two_d_size', [1, 100, 27])
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_round_trip_3d(two_d_size: int, axis: int, radius: np.ndarray, transformer: HankelTransform):
    dims = np.ones(3, np.int) * two_d_size
    dims[axis] = radius.size
    func = np.random.random(dims)
    ht = transformer.qdht(func, axis=axis)
    reconstructed = transformer.iqdht(ht, axis=axis)
    assert np.allclose(func, reconstructed)


@pytest.mark.parametrize('shape', smooth_shapes)
def test_round_trip_with_interpolation(shape: Callable,
                                       radius: np.ndarray,
                                       transformer: HankelTransform):
    # the function must be smoothish for interpolation
    # to work. Random every point doesn't work
    func = shape(radius)
    func_hr = transformer.to_transform_r(func)
    ht = transformer.qdht(func_hr)
    reconstructed_hr = transformer.iqdht(ht)
    reconstructed = transformer.to_original_r(reconstructed_hr)

    assert np.allclose(func, reconstructed, rtol=2e-4)


def test_original_r_k_grid():
    r_1d = np.linspace(0, 1, 10)
    k_1d = r_1d.copy()
    transformer = HankelTransform(order=0, max_radius=1, n_points=10)
    with pytest.raises(ValueError):
        _ = transformer.original_radial_grid
    with pytest.raises(ValueError):
        _ = transformer.original_k_grid

    transformer = HankelTransform(order=0, radial_grid=r_1d)
    # no error
    _ = transformer.original_radial_grid
    with pytest.raises(ValueError):
        _ = transformer.original_k_grid

    transformer = HankelTransform(order=0, k_grid=k_1d)
    # no error
    _ = transformer.original_k_grid
    with pytest.raises(ValueError):
        _ = transformer.original_radial_grid


def test_initialisation_errors():
    r_1d = np.linspace(0, 1, 10)
    k_1d = r_1d.copy()
    r_2d = np.repeat(r_1d[:, np.newaxis], repeats=5, axis=1)
    k_2d = r_2d.copy()
    with pytest.raises(ValueError):
        # missing any radius or k info
        HankelTransform(order=0)
    with pytest.raises(ValueError):
        # missing n_points
        HankelTransform(order=0, max_radius=1)
    with pytest.raises(ValueError):
        # missing max_radius
        HankelTransform(order=0, n_points=10)
    with pytest.raises(ValueError):
        # radial_grid and n_points
        HankelTransform(order=0, radial_grid=r_1d, n_points=10)
    with pytest.raises(ValueError):
        # radial_grid and max_radius
        HankelTransform(order=0, radial_grid=r_1d, max_radius=1)

    with pytest.raises(ValueError):
        # k_grid and n_points
        HankelTransform(order=0, k_grid=k_1d, n_points=10)
    with pytest.raises(ValueError):
        # k_grid and max_radius
        HankelTransform(order=0, k_grid=k_1d, max_radius=1)
    with pytest.raises(ValueError):
        # k_grid and r_grid
        HankelTransform(order=0, k_grid=k_1d, radial_grid=r_1d)

    with pytest.raises(AssertionError):
        HankelTransform(order=0, radial_grid=r_2d)
    with pytest.raises(AssertionError):
        HankelTransform(order=0, radial_grid=k_2d)

    # no error
    _ = HankelTransform(order=0, max_radius=1, n_points=10)
    _ = HankelTransform(order=0, radial_grid=r_1d)
    _ = HankelTransform(order=0, k_grid=k_1d)


@pytest.mark.parametrize('n', [10, 100, 512, 1024])
@pytest.mark.parametrize('max_radius', [0.1, 10, 20, 1e6])
def test_r_creation_equivalence(n: int, max_radius: float):
    transformer1 = HankelTransform(order=0, n_points=1024, max_radius=50)
    r = np.linspace(0, 50, 1024)
    transformer2 = HankelTransform(order=0, radial_grid=r)

    for key, val in transformer1.__dict__.items():
        if key == '_original_radial_grid':
            continue
        val2 = getattr(transformer2, key)
        if val is None:
            assert val2 is None
        else:
            assert np.allclose(val, val2)


@pytest.mark.parametrize('shape', smooth_shapes)
@pytest.mark.parametrize('order', orders)
def test_round_trip_r_interpolation(radius: np.ndarray, order: int, shape: Callable):
    transformer = HankelTransform(order=order, radial_grid=radius)

    # the function must be smoothish for interpolation
    # to work. Random every point doesn't work
    func = shape(radius)
    transform_func = transformer.to_transform_r(func)
    reconstructed_func = transformer.to_original_r(transform_func)
    assert np.allclose(func, reconstructed_func, rtol=1e-4)


@pytest.mark.parametrize('shape', smooth_shapes)
@pytest.mark.parametrize('order', orders)
def test_round_trip_k_interpolation(radius: np.ndarray, order: int, shape: Callable):
    k_grid = radius/10
    transformer = HankelTransform(order=order, k_grid=k_grid)

    # the function must be smoothish for interpolation
    # to work. Random every point doesn't work
    func = shape(k_grid)
    transform_func = transformer.to_transform_k(func)
    reconstructed_func = transformer.to_original_k(transform_func)
    assert np.allclose(func, reconstructed_func, rtol=1e-4)


@pytest.mark.parametrize('shape', smooth_shapes)
@pytest.mark.parametrize('order', orders)
@pytest.mark.parametrize('axis', [0, 1])
def test_round_trip_r_interpolation_2d(radius: np.ndarray, order: int, shape: Callable, axis: int):
    transformer = HankelTransform(order=order, radial_grid=radius)

    # the function must be smoothish for interpolation
    # to work. Random every point doesn't work
    dims_amplitude = np.ones(2, np.int)
    dims_amplitude[1-axis] = 10
    amplitude = np.random.random(dims_amplitude)
    dims_radius = np.ones(2, np.int)
    dims_radius[axis] = len(radius)
    func = np.reshape(shape(radius), dims_radius) * np.reshape(amplitude, dims_amplitude)
    transform_func = transformer.to_transform_r(func, axis=axis)
    reconstructed_func = transformer.to_original_r(transform_func, axis=axis)
    assert np.allclose(func, reconstructed_func, rtol=1e-4)


@pytest.mark.parametrize('shape', smooth_shapes)
@pytest.mark.parametrize('order', orders)
@pytest.mark.parametrize('axis', [0, 1])
def test_round_trip_k_interpolation_2d(radius: np.ndarray, order: int, shape: Callable, axis: int):
    k_grid = radius/10
    transformer = HankelTransform(order=order, k_grid=k_grid)

    # the function must be smoothish for interpolation
    # to work. Random every point doesn't work
    dims_amplitude = np.ones(2, np.int)
    dims_amplitude[1-axis] = 10
    amplitude = np.random.random(dims_amplitude)
    dims_k = np.ones(2, np.int)
    dims_k[axis] = len(radius)
    func = np.reshape(shape(k_grid), dims_k) * np.reshape(amplitude, dims_amplitude)
    transform_func = transformer.to_transform_k(func, axis=axis)
    reconstructed_func = transformer.to_original_k(transform_func, axis=axis)
    assert np.allclose(func, reconstructed_func, rtol=1e-4)


# -------------------
# Test known HT pairs
# -------------------

@pytest.mark.parametrize('a', [1, 0.7, 0.1])
def test_jinc(transformer: HankelTransform, a: float):
    f = generalised_jinc(transformer.r, a, transformer.order)
    expected_ht = generalised_top_hat(transformer.v, a, transformer.order)
    actual_ht = transformer.qdht(f)
    error = np.mean(np.abs(expected_ht-actual_ht))
    assert error < 1e-3


@pytest.mark.parametrize('two_d_size', [1, 100, 27])
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('a', [1, 0.7, 0.1])
def test_jinc2d(transformer: HankelTransform, a: float, axis: int, two_d_size: int):
    f = generalised_jinc(transformer.r, a, transformer.order)
    second_axis = np.outer(np.linspace(0, 6, two_d_size), f)
    expected_ht = generalised_top_hat(transformer.v, a, transformer.order)
    if axis == 0:
        f_array = np.outer(f, second_axis)
        expected_ht_array = np.outer(expected_ht, second_axis)
    else:
        f_array = np.outer(second_axis, f)
        expected_ht_array = np.outer(second_axis, expected_ht)
    actual_ht = transformer.qdht(f_array, axis=axis)
    error = np.mean(np.abs(expected_ht_array-actual_ht))
    assert error < 1e-3


@pytest.mark.parametrize('a', [2, 5, 10])
def test_gaussian(a: float, radius: np.ndarray):
    # Note the definition in Guizar-Sicairos varies by 2*pi in
    # both scaling of the argument (so use kr rather than v) and
    # scaling of the magnitude.
    transformer = HankelTransform(order=0, radial_grid=radius)
    f = np.exp(-a ** 2 * transformer.r ** 2)
    expected_ht = 2*np.pi*(1 / (2 * a**2)) * np.exp(-transformer.kr**2 / (4 * a**2))
    actual_ht = transformer.qdht(f)
    assert np.allclose(expected_ht, actual_ht)


@pytest.mark.parametrize('a', [2, 5, 10])
def test_inverse_gaussian(a: float, radius: np.ndarray):
    # Note the definition in Guizar-Sicairos varies by 2*pi in
    # both scaling of the argument (so use kr rather than v) and
    # scaling of the magnitude.
    transformer = HankelTransform(order=0, radial_grid=radius)
    ht = 2*np.pi*(1 / (2 * a**2)) * np.exp(-transformer.kr**2 / (4 * a**2))
    actual_f = transformer.iqdht(ht)
    expected_f = np.exp(-a ** 2 * transformer.r ** 2)
    assert np.allclose(expected_f, actual_f)


@pytest.mark.parametrize('axis', [0, 1])
def test_gaussian_2d(axis: int, radius: np.ndarray):
    # Note the definition in Guizar-Sicairos varies by 2*pi in
    # both scaling of the argument (so use kr rather than v) and
    # scaling of the magnitude.
    transformer = HankelTransform(order=0, radial_grid=radius)
    a = np.linspace(2, 10)
    dims_a = np.ones(2, np.int)
    dims_a[1-axis] = len(a)
    dims_r = np.ones(2, np.int)
    dims_r[axis] = len(transformer.r)
    a_reshaped = np.reshape(a, dims_a)
    r_reshaped = np.reshape(transformer.r, dims_r)
    kr_reshaped = np.reshape(transformer.kr, dims_r)
    f = np.exp(-a_reshaped**2 * r_reshaped**2)
    expected_ht = 2*np.pi*(1 / (2 * a_reshaped**2)) * np.exp(-kr_reshaped**2 / (4 * a_reshaped**2))
    actual_ht = transformer.qdht(f, axis=axis)
    assert np.allclose(expected_ht, actual_ht)


@pytest.mark.parametrize('axis', [0, 1])
def test_inverse_gaussian_2d(axis: int, radius: np.ndarray):
    # Note the definition in Guizar-Sicairos varies by 2*pi in
    # both scaling of the argument (so use kr rather than v) and
    # scaling of the magnitude.
    transformer = HankelTransform(order=0, radial_grid=radius)
    a = np.linspace(2, 10)
    dims_a = np.ones(2, np.int)
    dims_a[1-axis] = len(a)
    dims_r = np.ones(2, np.int)
    dims_r[axis] = len(transformer.r)
    a_reshaped = np.reshape(a, dims_a)
    r_reshaped = np.reshape(transformer.r, dims_r)
    kr_reshaped = np.reshape(transformer.kr, dims_r)
    ht = 2*np.pi*(1 / (2 * a_reshaped**2)) * np.exp(-kr_reshaped**2 / (4 * a_reshaped**2))
    actual_f = transformer.iqdht(ht, axis=axis)
    expected_f = np.exp(-a_reshaped ** 2 * r_reshaped ** 2)
    assert np.allclose(expected_f, actual_f)


@pytest.mark.parametrize('a', [2, 1, 0.1])
def test_1_over_r2_plus_z2(a: float):
    # Note the definition in Guizar-Sicairos varies by 2*pi in
    # both scaling of the argument (so use kr rather than v) and
    # scaling of the magnitude.
    transformer = HankelTransform(order=0, n_points=1024, max_radius=50)
    f = 1 / (transformer.r**2 + a**2)
    # kn cannot handle complex arguments, so a must be real
    expected_ht = 2 * np.pi * scipy_bessel.kn(0, a * transformer.kr)
    actual_ht = transformer.qdht(f)
    # These tolerances are pretty loose, but there seems to be large
    # error here
    assert np.allclose(expected_ht, actual_ht, rtol=0.1, atol=0.01)
    error = np.mean(np.abs(expected_ht - actual_ht))
    assert error < 4e-3


def sinc(x):
    return np.sin(x) / x


# noinspection DuplicatedCode
@pytest.mark.parametrize('p', [1, 4])
def test_sinc(p):
    """Tests from figure 1 of
        *"Computation of quasi-discrete Hankel transforms of the integer
        order for propagating optical wave fields"*
        Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega
        J. Opt. Soc. Am. A **21** (1) 53-58 (2004)
        """
    transformer = HankelTransform(p, max_radius=3, n_points=256)
    v = transformer.v
    gamma = 5
    func = sinc(2 * np.pi * gamma * transformer.r)
    expected_ht = np.zeros_like(func)

    expected_ht[v < gamma] = (v[v < gamma]**p * np.cos(p * np.pi / 2)
                              / (2*np.pi*gamma * np.sqrt(gamma**2 - v[v < gamma]**2)
                                 * (gamma + np.sqrt(gamma**2 - v[v < gamma]**2))**p))
    expected_ht[v >= gamma] = (np.sin(p * np.arcsin(gamma/v[v >= gamma]))
                               / (2*np.pi*gamma * np.sqrt(v[v >= gamma]**2 - gamma**2)))
    ht = transformer.qdht(func)

    # use the same error measure as the paper
    dynamical_error = 20 * np.log10(np.abs(expected_ht-ht) / np.max(ht))
    not_near_gamma = np.logical_or(v > gamma*1.25,
                                   v < gamma*0.75)
    assert np.all(dynamical_error < -10)
    assert np.all(dynamical_error[not_near_gamma] < -35)
*/
