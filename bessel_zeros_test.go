package gohank

import (
	"encoding/csv"
	"fmt"
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

const J_ZEROS string = `#
#k	J_0(x)	J_1(x)	J_2(x)	J_3(x)	J_4(x)	J_5(x)
1	2.4048	3.8317	5.1356	6.3802	7.5883	8.7715
2	5.5201	7.0156	8.4172	9.7610	11.0647	12.3386
3	8.6537	10.1735	11.6198	13.0152	14.3725	15.7002
4	11.7915	13.3237	14.7960	16.2235	17.6160	18.9801
5	14.9309	16.4706	17.9598	19.4094	20.8269	22.2178`

const JP_ZEROS string = `#
#k	J_0'(x)	J_1'(x)	J_2'(x)	J_3'(x)	J_4'(x)	J_5'(x)
1	0.0000	1.8412	3.0542	4.2012	5.3175	6.4156
2	3.8317	5.3314	6.7061	8.0152	9.2824	10.5199
3	7.0156	8.5363	9.9695	11.3459	12.6819	13.9872
4	10.1735	11.7060	13.1704	14.5858	15.9641	17.3128
5	13.3237	14.8636	16.3475	17.7887	19.1960	20.5755`

//	16.4706

const Y_ZEROS = `#
0.89357697	3.95767842	7.08605106	10.22234504	13.36109747
2.19714133	5.42968104	8.59600587	11.74915483	14.89744213
3.38424177	6.79380751	10.02347798	13.20998671	16.37896656
4.52702466	8.09755376	11.39646674	14.62307774	17.81845523
5.64514789	9.36162062	12.73014447	15.99962709	19.22442896`

const YP_ZEROS = `#
2.19714133	5.42968104	8.59600587	11.74915483	14.89744213
3.68302286	6.94149995	10.12340466	13.28575816	16.44005801
5.00258293	8.3507247	11.57419547	14.76090931	17.93128594
6.25363321	9.69878798	12.97240905	16.1904472	19.38238845
7.46492174	11.00516915	14.33172352	17.58443602	20.80106234`

var all_zeros [][][]float64

const maxOrder = 5

func getRecords(str string) [][]string {
	r := csv.NewReader(strings.NewReader(str))
	r.Comma = '\t'
	r.Comment = '#'

	records, err := r.ReadAll()
	if err != nil {
		panic(err)
	}
	return records
}

func parseZerosPython(str string) [][]float64 {
	records := getRecords(str)
	floats := make([][]float64, maxOrder)
	for order := 0; order < maxOrder; order++ {
		floats[order] = make([]float64, len(records))
		for i := 0; i < len(records); i++ {
			var err error
			floats[order][i], err = strconv.ParseFloat(records[order][i], 64)
			if err != nil {
				panic(err)
			}
		}
	}
	return floats
}

func parseZerosWA(str string) [][]float64 {
	records := getRecords(str)
	floats := make([][]float64, maxOrder)
	for order := 0; order < maxOrder; order++ {
		floats[order] = make([]float64, len(records))
		for i := 0; i < len(records); i++ {
			// order+ 1 as the first col is an index
			col := order + 1
			var err error
			floats[order][i], err = strconv.ParseFloat(records[i][col], 64)
			if err != nil {
				panic(err)
			}
		}
	}
	return floats
}

func init() {
	var j_zeros, jp_zeros, y_zeros, yp_zeros [][]float64
	j_zeros = parseZerosWA(J_ZEROS)
	jp_zeros = parseZerosWA(JP_ZEROS)
	y_zeros = parseZerosPython(Y_ZEROS)
	yp_zeros = parseZerosPython(YP_ZEROS)
	all_zeros = [][][]float64{j_zeros, y_zeros, jp_zeros, yp_zeros}
}

func TestAgainstHardCodedZeros(t *testing.T) {
	for funType := 0; funType <= int(YP); funType++ {
		these_zeros := all_zeros[funType]
		for order := range these_zeros {
			t.Run(fmt.Sprint(order), func(t *testing.T) {
				expected := these_zeros[order]
				actual := besselZeros(BesselFunType(funType), order, len(expected), 1e-6)
				assert.InDeltaSlice(t, expected, actual, 1e-4)
			})
		}
	}
}

func TestEvaluationAtZeroJ(t *testing.T) {
	for order := 0; order < 20; order++ {
		t.Run(fmt.Sprint(order), func(t *testing.T) {
			zeros := besselZeros(J, order, 100, 1e-6)
			for _, v := range zeros {
				assert.InDelta(t, 0, math.Jn(order, v), 1e-6)
			}
		})
	}
}

func TestEvaluationAtZeroY(t *testing.T) {
	for order := 0; order < 20; order++ {
		t.Run(fmt.Sprint(order), func(t *testing.T) {
			zeros := besselZeros(Y, order, 100, 1e-6)
			for _, v := range zeros {
				assert.InDelta(t, 0, math.Yn(order, v), 1e-6)
			}
		})
	}
}
