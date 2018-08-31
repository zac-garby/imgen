package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/NOX73/go-neural"
	"github.com/NOX73/go-neural/engine"
)

var (
	// Iterations is the number of training iterations to do
	Iterations = flag.Int("train-iters", 200, "the amount of training iterations")

	// OutGenerations is the number of output iterations to do
	OutGenerations = flag.Int("out-iters", 8, "the amout of output iterations")

	// LearnRate is the rate at which the network learns
	LearnRate = flag.Float64("rate", 0.1, "the speed at which the network learns")

	// OutWidth is the width of the generated image
	OutWidth = flag.Int("width", 32, "the width of the generated image")

	// OutHeight is the height of the generated image
	OutHeight = flag.Int("height", 32, "the height of the generated image")

	// In is the name of the input file
	In = flag.String("in", "ref.png", "the name of the input file")

	// Out is the name of the output file
	Out = flag.String("out", "out.png", "the name of the output file")

	// NeighbourRadius is the radius inside which a pixel is considered a neighbour
	NeighbourRadius = flag.Int("n-rad", 2, "the radius inside which a pixel is considered a neighbour")
)

func main() {
	flag.Parse()

	rand.Seed(time.Now().UnixNano())

	reader, err := os.Open(*In)
	if err != nil {
		log.Fatal(err)
	}

	m, _, err := image.Decode(reader)
	if err != nil {
		log.Fatal(err)
	}

	bounds := m.Bounds()

	n := neural.NewNetwork(neighbourNum()*3, []int{neighbourNum()*2, neighbourNum()*2, 3})
	n.RandomizeSynapses()

	engine := engine.New(n)
	engine.Start()

	fmt.Println("started training...")

	for i := 0; i < *Iterations; i++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
				ins := neighbours(m, bounds, x, y)
				ar, ag, ab := at(m, bounds, x, y)

				engine.Learn(ins, []float64{ar, ag, ab}, *LearnRate)
			}
		}
		fmt.Printf("finished iteration %d/%d    \r", i+1, *Iterations)
	}

	fmt.Println("\ndone! generating output image...")

	imgRect := image.Rect(0, 0, *OutWidth, *OutHeight)
	img := image.NewRGBA(imgRect)

	for x := 0; x < *OutWidth; x++ {
		for y := 0; y < *OutHeight; y++ {
			img.SetRGBA(x, y, color.RGBA{
				uint8(rand.Float64() * 255),
				uint8(rand.Float64() * 255),
				uint8(rand.Float64() * 255),
				255,
			})
		}
	}

	for i := 0; i < *OutGenerations; i++ {
		newImg := image.NewRGBA(imgRect)

		for x := 0; x < *OutWidth; x++ {
			for y := 0; y < *OutHeight; y++ {
				ins := neighbours(img, imgRect, x, y)
				outs := engine.Calculate(ins)
				newImg.SetRGBA(x, y, color.RGBA{
					uint8(outs[0] * 255),
					uint8(outs[1] * 255),
					uint8(outs[2] * 255),
					255,
				})
			}
		}

		img = newImg
		fmt.Printf("finished iteration %d/%d    \r", i+1, *OutGenerations)
	}

	out, err := os.Create(*Out)
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
	neighbours := []float64{}

	for nx := x - *NeighbourRadius; nx < x+*NeighbourRadius+1; nx++ {
		for ny := y - *NeighbourRadius; ny < y+*NeighbourRadius+1; ny++ {
			if nx == x && ny == y {
				continue
			}

			r, g, b := at(m, bounds, nx, ny)
			neighbours = append(neighbours, r, g, b)
		}
	}

	return neighbours
}

func neighbourNum() int {
	return *NeighbourRadius**NeighbourRadius*4 + *NeighbourRadius*4
}
