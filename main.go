package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/NOX73/go-neural"
	"github.com/NOX73/go-neural/learn"
)

const (
	// Iterations is the number of training iterations to do
	Iterations = 100

	// OutGenerations is the number of output iterations to do
	OutGenerations = 2

	// LearnRate is the rate at which the network learns
	LearnRate = 0.1

	// OutWidth is the width of the generated image
	OutWidth = 16

	// OutHeight is the height of the generated image
	OutHeight = 16
)

func main() {
	rand.Seed(time.Now().UnixNano())

	reader, err := os.Open("ref.png")
	if err != nil {
		log.Fatal(err)
	}

	m, _, err := image.Decode(reader)
	if err != nil {
		log.Fatal(err)
	}

	bounds := m.Bounds()

	n := neural.NewNetwork(8*3, []int{16, 16, 3})
	n.RandomizeSynapses()

	fmt.Println("started training...")

	for i := 0; i < Iterations; i++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
				ins := neighbours(m, bounds, x, y)
				ar, ag, ab := at(m, bounds, x, y)

				learn.Learn(n, ins, []float64{ar, ag, ab}, LearnRate)
			}
		}
		fmt.Printf("finished iteration %d/%d    \r", i+1, Iterations)
	}

	fmt.Println("\ndone! generating output image...")

	imgRect := image.Rect(0, 0, OutWidth, OutHeight)
	img := image.NewRGBA(imgRect)

	for x := 0; x < OutWidth; x++ {
		for y := 0; y < OutHeight; y++ {
			img.SetRGBA(x, y, color.RGBA{
				uint8(rand.Float64() * 255),
				uint8(rand.Float64() * 255),
				uint8(rand.Float64() * 255),
				255,
			})
		}
	}

	for i := 0; i < OutGenerations; i++ {
		newImg := image.NewRGBA(imgRect)

		for x := 0; x < OutWidth; x++ {
			for y := 0; y < OutHeight; y++ {
				ins := neighbours(img, imgRect, x, y)
				outs := n.Calculate(ins)
				newImg.SetRGBA(x, y, color.RGBA{
					uint8(outs[0] * 255),
					uint8(outs[1] * 255),
					uint8(outs[2] * 255),
					255,
				})
			}
		}

		img = newImg
		fmt.Printf("finished iteration %d/%d    \r", i+1, OutGenerations)
	}

	out, err := os.Create("./out.png")
	if err != nil {
		log.Fatal(err)
	}

	if err := png.Encode(out, img); err != nil {
		log.Fatal(err)
	}

	fmt.Println("\ndone!")
}

func at(m image.Image, bounds image.Rectangle, x, y int) (r float64, g float64, b float64) {
	if x < bounds.Min.X || y < bounds.Min.Y || x >= bounds.Max.X || y >= bounds.Max.Y {
		return 0, 0, 0
	}

	ri, gi, bi, _ := m.At(x, y).RGBA()
	r = float64(ri) / 65535
	g = float64(gi) / 65535
	b = float64(bi) / 65535
	return
}

func neighbours(m image.Image, bounds image.Rectangle, x, y int) []float64 {
	c0r, c0g, c0b := at(m, bounds, x-1, y-1)
	c1r, c1g, c1b := at(m, bounds, x, y-1)
	c2r, c2g, c2b := at(m, bounds, x+1, y-1)
	c3r, c3g, c3b := at(m, bounds, x-1, y)
	c4r, c4g, c4b := at(m, bounds, x+1, y)
	c5r, c5g, c5b := at(m, bounds, x-1, y+1)
	c6r, c6g, c6b := at(m, bounds, x, y+1)
	c7r, c7g, c7b := at(m, bounds, x+1, y+1)

	return []float64{
		c0r, c0g, c0b,
		c1r, c1g, c1b,
		c2r, c2g, c2b,
		c3r, c3g, c3b,
		c4r, c4g, c4b,
		c5r, c5g, c5b,
		c6r, c6g, c6b,
		c7r, c7g, c7b,
	}
}
