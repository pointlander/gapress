// Copyright 2017 The GAPress Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/nfnt/resize"
	"github.com/pointlander/compress"
	ga "github.com/pointlander/go-galib"
)

const (
	testImage = "images/image01.png"
	scale     = 8
)

func Gray(input image.Image) *image.Gray {
	bounds := input.Bounds()
	output := image.NewGray(bounds)
	width, height := bounds.Max.X, bounds.Max.Y
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			r, g, b, _ := input.At(x, y).RGBA()
			output.SetGray(x, y, color.Gray{uint8((float64(r)+float64(g)+float64(b))/(3*256) + .5)})
		}
	}
	return output
}

type Initializer struct{}

func (i *Initializer) InitPop(first ga.GAGenome, popsize int) (pop []ga.GAGenome) {
	pop = make([]ga.GAGenome, popsize)
	for x := 0; x < popsize; x++ {
		pop[x] = first.Copy()
	}
	return pop
}

func (i *Initializer) String() string { return "Initializer" }

type Fitness struct {
	scores int
	img    *image.Gray
}

// Boring fitness/score function.
func (f *Fitness) score(g *ga.GAIntGenome) float64 {
	var total int
	for _, c := range g.Gene {
		total += c
	}
	f.scores++
	if total < 0 {
		total = -total
	}

	pixels := make([]byte, len(f.img.Pix))
	copy(pixels, f.img.Pix)
	for i, pixel := range pixels {
		pix := int(pixel) + g.Gene[i]
		if pix > 255 {
			pix = 255
		} else if pix < 0 {
			pix = 0
		}
		pixels[i] = uint8(pix)
	}

	output := &bytes.Buffer{}
	compress.Mark1Compress16(pixels, output)
	_ = total
	fitness := float64(output.Len())
	return fitness * fitness
}

func main() {
	rand.Seed(time.Now().UTC().UnixNano())
	fit := &Fitness{}

	file, err := os.Open(testImage)
	if err != nil {
		log.Fatal(err)
	}

	info, err := file.Stat()
	if err != nil {
		log.Fatal(err)
	}
	name := info.Name()
	name = name[:strings.Index(name, ".")]

	input, _, err := image.Decode(file)
	if err != nil {
		log.Fatal(err)
	}
	file.Close()

	width, height := input.Bounds().Max.X, input.Bounds().Max.Y
	width, height = width/scale, height/scale
	input = resize.Resize(uint(width), uint(height), input, resize.NearestNeighbor)
	fit.img = Gray(input)

	file, err = os.Create(name + ".png")
	if err != nil {
		log.Fatal(err)
	}

	err = png.Encode(file, fit.img)
	if err != nil {
		log.Fatal(err)
	}
	file.Close()

	m := ga.NewMultiMutator()
	msh := new(ga.GAShiftMutator)
	msw := new(ga.GASwitchMutator)
	mr := new(ga.GAMutatorRandom)
	m.Add(msh)
	m.Add(msw)
	m.Add(mr)

	param := ga.GAParameter{
		Initializer: new(Initializer),
		Selector:    ga.NewGATournamentSelector(0.7, 5),
		Breeder:     new(ga.GA2PointBreeder),
		Mutator:     m,
		PMutate:     0.5,
		PBreed:      0.7}

	gao := ga.NewGA(param)

	genome := ga.NewIntGenome(make([]int, len(fit.img.Pix)), fit.score, -10, 10)

	output := &bytes.Buffer{}
	compress.Mark1Compress16(fit.img.Pix, output)
	inputSize := output.Len()

	gao.Init(10, genome) //Total population
	gao.OptimizeUntil(func(best ga.GAGenome) bool {
		pixels, gene := make([]byte, len(fit.img.Pix)), best.(*ga.GAIntGenome).Gene
		copy(pixels, fit.img.Pix)
		for i, pixel := range pixels {
			pix := int(pixel) + gene[i]
			if pix > 255 {
				pix = 255
			} else if pix < 0 {
				pix = 0
			}
			pixels[i] = uint8(pix)
		}
		output := &bytes.Buffer{}
		compress.Mark1Compress16(pixels, output)
		fmt.Println(float64(output.Len()) / float64(inputSize))
		return best.Score() == 0
	})
	gao.PrintTop(10)

	fmt.Printf("Calls to score = %d\n", fit.scores)
	fmt.Printf("%s\n", m.Stats())
}
