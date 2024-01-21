package gohank

import (
	"math"
	_ "math/cmplx"
)

const pi = math.Pi

type BesselFunType int

const (
	J BesselFunType = iota
	Y
	JP
	YP
)

func besselZeros(funcType BesselFunType, order int, n int, e float64) (z []float64) {
	//BESSEL_ZEROS: Finds the first n zeros of a bessel function
	//
	//	z = bessel_zeros(d, a, n, e)
	//
	//	z	=	zeros of the bessel function
	//	d	=	Bessel function type:
	//			1:	Ja
	//			2:	Ya
	//			3:	Ja'
	//			4:	Ya'
	//	a	=	Bessel order (a>=0)
	//	n	=	Number of zeros to find
	//	e	=	Relative error in root
	//
	//	This function uses the routine described in:
	//		"An Algorithm with ALGOL 60 Program for the Computation of the
	//		zeros of the Ordinary Bessel Functions and those of their
	//		Derivatives".
	//		N. M. Temme
	//		Journal of Computational Physics, 32, 270-279 (1979)
	//
	// Translated from Adam Wyatt's Matlab version
	a := float64(order)
	funcType++
	z = make([]float64, n)

	aa := math.Pow(a, 2)
	mu := 4 * aa
	mu2 := math.Pow(mu, 2)
	mu3 := math.Pow(mu, 3)
	mu4 := math.Pow(mu, 4)

	var p, p0, p1, q1 float64
	if funcType < 3 {
		p = 7*mu - 31
		p0 = mu - 1

		if (1 + p) == p {
			p1 = 0
			q1 = 0
		} else {
			p1 = 4 * (253*mu2 - 3722*mu + 17869) * p0 / (15 * p)
			q1 = 1.6 * (83*mu2 - 982*mu + 3779) / p
		}
	} else {
		p = 7*mu2 + 82*mu - 9
		p0 = mu + 3
		if (p + 1) == 1 {
			p1 = 0
			q1 = 0
		} else {
			p1 = (4048*mu4 + 131264*mu3 - 221984*mu2 - 417600*mu + 1012176) / (60 * p)
			q1 = 1.6 * (83*mu3 + 2075*mu2 - 3039*mu + 3537) / p
		}
	}

	var t float64
	if (funcType == 1) || (funcType == 4) {
		t = .25
	} else {
		t = .75
	}
	tt := 4 * t

	var pp1, qq1 float64
	if funcType < 3 {
		pp1 = 5. / 48.
		qq1 = -5. / 36.
	} else {
		pp1 = -7. / 48.
		qq1 = 35. / 288.
	}

	y := .375 * pi
	var bb float64
	if a >= 3 {
		bb = math.Pow(a, (-2.0 / 3.0))
	} else {
		bb = 1
	}
	var a1 int = 3*int(a) - 8
	// psi = (.5*a + .25)*pi;

	for s := 1; s <= n; s++ {
		var x, w float64
		var j int
		if (order == 0) && (s == 1) && (funcType == 3) {
			x = 0
			j = 0
		} else {
			if s >= a1 {
				b := (float64(s) + 0.5*a - t) * pi
				c := .015625 / (math.Pow(b, 2))
				x = b - .125*(p0-p1*c)/(b*(1-q1*c))
			} else {
				if s == 1 {
					switch funcType {
					case (1):
						x = -2.33811
					case (2):
						x = -1.17371
					case (3):
						x = -1.01879
					case 4:
						x = -2.29444
					default:
						panic("not implemented")
					}
				} else {
					x = y * (4*float64(s) - tt)
					v := math.Pow(x, -2)
					x = -math.Pow(x, (2.0/3.0)) * (1 + v*(pp1+qq1*v))
				}
				u := x * bb
				v := fi(2.0 / 3.0 * math.Pow(-u, 1.5))
				w = 1 / math.Cos(v)
				xx := 1 - math.Pow(w, 2)
				c := math.Sqrt(u / xx)
				if funcType < 3 {
					x = w * (a + c*(-5/u-c*(6-10/xx))/(48*a*u))
				} else {
					x = w * (a + c*(7/u+c*(18-14/xx))/(48*a*u))
				}
			}
			j = 0

			for (j == 0) || ((j < 5) && (math.Abs(w/x) > e)) {
				xx := math.Pow(x, 2)
				x4 := math.Pow(x, 4)
				a2 := aa - xx
				r0 := bessr(funcType, order, x)
				j = j + 1
				var q, u float64
				if funcType < 3 {
					u = r0
					w = 6 * x * (2*a + 1)
					p = (1 - 4*a2) / w
					q = (4*(xx-mu) - 2 - 12*a) / w
				} else {
					u = -xx * r0 / a2
					v := 2 * x * a2 / (3 * (aa + xx))
					w = 64 * math.Pow(a2, 3)
					q = 2 * v * (1 + mu2 + 32*mu*xx + 48*x4) / w
					p = v * (1 + (40*mu*xx+48*x4-mu2)/w)
				}
				w = u * (1 + p*r0) / (1 + q*r0)
				x = x + w
			}
			z[s-1] = x
		}
	}
	return
}

func fi(y float64) (FI float64) {
	c1 := 1.570796
	if y == 0 {
		FI = 0
	} else if y > 1e5 {
		FI = c1
	} else {
		var p float64
		if y < 1 {
			p = math.Pow(3*y, (1.0 / 3.0))
			pp := math.Pow(p, 2)
			p = p * (1 + pp*(pp*(27-2*pp)-210)/1575)
		} else {
			p = 1 / (y + c1)
			pp := math.Pow(p, 2)
			p = c1 - p*(1+pp*(2310+pp*(3003+pp*(4818+pp*(8591+pp*16328))))/3465)
		}
		pp := math.Pow((y + p), 2)
		r := (p - math.Atan(p+y)) / pp
		FI = p - (1+pp)*r*(1+r/(p+y))
	}
	return
}

func bessr(funType BesselFunType, order int, x float64) (Jr float64) {
	Jr = 0.0
	a := float64(order)
	switch funType {
	case 1:
		Jr = math.Jn(order, x) / math.Jn(order+1, x)
	case 2:
		Jr = math.Yn(order, x) / math.Yn(order+1, x)
	case 3:
		Jr = a/x - math.Jn(order+1, x)/math.Jn(order, x)
	case 4:
		Jr = a/x - math.Yn(order+1, x)/math.Yn(order, x)
	}
	return
}
