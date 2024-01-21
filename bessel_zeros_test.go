package gohank

import (
	"encoding/csv"
	"fmt"
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

var j_zeros, jp_zeros [][]float64

const maxOrder = 5

func parseZeros(str string) [][]float64 {
	r := csv.NewReader(strings.NewReader(str))
	r.Comma = '\t'
	r.Comment = '#'

	records, err := r.ReadAll()
	if err != nil {
		panic(err)
	}

	floats := make([][]float64, maxOrder)
	for order := 0; order < maxOrder; order++ {
		floats[order] = make([]float64, len(records))
		for i := 0; i < len(records); i++ {
			// order+ 1 as the first col is an index
			col := order + 1
			floats[order][i], err = strconv.ParseFloat(records[i][col], 64)
			if err != nil {
				panic(err)
			}
		}
	}
	return floats
}

func init() {
	j_zeros = parseZeros(J_ZEROS)
	jp_zeros = parseZeros(JP_ZEROS)
}

func TestJAgainstHardCodedZeros(t *testing.T) {
	for order := range j_zeros {
		t.Run(fmt.Sprint(order), func(t *testing.T) {
			expected := j_zeros[order]
			actual := besselZeros(J, order, len(expected), 1e-6)
			assert.InDeltaSlice(t, expected, actual, 1e-4)
		})
	}
}

func TestJPAgainstHardCodedZeros(t *testing.T) {
	for order := range j_zeros {
		t.Run(fmt.Sprint(order), func(t *testing.T) {
			expected := jp_zeros[order]
			actual := besselZeros(JP, order, len(expected), 1e-6)
			assert.InDeltaSlice(t, expected, actual, 1e-4)
		})

	}
}
