package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"os"
	"regexp"
	"strconv"
	"time"

	"golang.org/x/exp/rand"
	"maze.io/x/math32"

	"code.sajari.com/word2vec"
	"gonum.org/v1/gonum/stat/sampleuv"
)

const LEXICON_PATH = "lexique-grammalecte-fr-v7.0.csv"
const WORD2VEC_PATH = "frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"
const CEMANTIX_URL = "https://cemantix.certitudes.org"
const COEF_SUM = 0
const COEF_PROD = 1

func loadBinary(path string) *word2vec.Model {
	r, err := os.Open(path)
	if err != nil {
		log.Fatalf("error opening file: %v", err)
	}
	model, err := word2vec.FromReader(r)
	if err != nil {
		log.Fatalf("error loading model: %v", err)
	}
	return model
}

func readCsvFile(filePath string) [][]string {
	f, err := os.Open(filePath)
	if err != nil {
		log.Fatal("Unable to read input file "+filePath, err)
	}
	defer f.Close()
	csvReader := csv.NewReader(f)
	csvReader.Comma = '\t' // Use tab-delimited instead of comma
	records, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal("Unable to parse file as CSV for "+filePath, err)
	}
	return records
}

func processCSV(records [][]string, model *word2vec.Model) ([]string, []float32, error) {
	modelWords := make(map[string]bool)
	for _, word := range model.Words() {
		modelWords[word] = true
	}
	var words []string
	var freqs []float32
	for _, record := range records[1:] {
		if record[2] != record[3] {
			continue
		}
		match, _ := regexp.MatchString("nom|v[123]|adj", record[4])
		if !match {
			continue
		}
		new, ok := modelWords[record[3]]
		if !ok {
			continue
		}
		if new {
			modelWords[record[3]] = false
		} else {
			continue
		}
		words = append(words, record[3])
		f, err := strconv.ParseFloat(record[18], 64)
		if err != nil {
			return nil, nil, err
		}
		freqs = append(freqs, float32(f))
	}
	return words, freqs, nil
}

type Post struct {
	Num     int     `json:"num,omitempty"`
	Score   float32 `json:"score,omitempty"`
	Solvers int     `json:"solvers,omitempty"`
	Error   string  `json:"error,omitempty"`
}

func getScore(word string) (float32, error) {
	// curl 'https://cemantix.certitudes.org/score' -X POST -H 'Content-Type: application/x-www-form-urlencoded' -H 'Origin: https://cemantix.certitudes.org' --data-raw 'word=est' -i
	data := fmt.Sprintf(`word=%s`, word)
	body := []byte(data)
	url := fmt.Sprintf("%s/score", CEMANTIX_URL)
	r, err := http.NewRequest("POST", url, bytes.NewBuffer(body))
	if err != nil {
		return 0, err
	}
	r.Header.Add("Content-Type", "application/x-www-form-urlencoded")
	r.Header.Add("Origin", CEMANTIX_URL)
	client := &http.Client{}
	res, err := client.Do(r)
	if err != nil {
		return 0, err
	}
	if res.StatusCode >= 400 {
		return 0, fmt.Errorf("error: %s", res.Status)
	}
	defer res.Body.Close()
	post := &Post{}
	err = json.NewDecoder(res.Body).Decode(post)
	if err != nil {
		return 0, err
	}
	if post.Error != "" {
		return 0, fmt.Errorf("error: %s", post.Error)
	}
	return post.Score, nil
}

func makeWeights(freqs []float32, sums []float32, prods []float32) []float64 {
	newar := make([]float64, len(freqs))
	var i int
	sum := float64(0.0)
	for i = range freqs {
		newar[i] = float64(freqs[i]) * (COEF_SUM/(float64(sums[i])) + COEF_PROD/(float64(prods[i])))
		sum += newar[i]
	}
	for i = range newar {
		newar[i] /= sum
	}
	return newar
}

func computeSimilarities(word string, words []string, model *word2vec.Model) ([]float32, error) {
	pairs := [][2]word2vec.Expr{}
	expr := word2vec.Expr{word: 1}
	for _, w := range words {
		e := word2vec.Expr{w: 1}
		pairs = append(pairs, [2]word2vec.Expr{expr, e})
	}
	res, err := model.Coses(pairs)
	return res, err
}

func step(words []string, model *word2vec.Model, freqs []float32, sums []float32, prods []float32) (int, bool, error) {
	weights64 := makeWeights(freqs, sums, prods)
	sampler := sampleuv.NewWeighted(weights64, nil)
	index, _ := sampler.Take()
	word := words[index]
	score, err := getScore(word)
	if err != nil {
		return index, false, err
	}
	fmt.Println(word, score)
	if score == 1 {
		return index, true, nil
	}
	similarities, err := computeSimilarities(word, words, model)
	if err != nil {
		return index, false, err
	}
	for i, sim := range similarities {
		dist := math32.Abs(score - sim)
		if dist == 0 {
			_score, err := getScore(word)
			fmt.Println("--> TRY 0 DIST:", word, _score)
			if err == nil && _score == 1 {
				return 0, true, nil
			}
		}
		sums[i] += dist
		prods[i] *= dist
	}
	return index, false, nil
}

func main() {
	model := loadBinary(WORD2VEC_PATH)
	records := readCsvFile(LEXICON_PATH)
	words, freqs, err := processCSV(records, model)
	if err != nil {
		log.Fatal(err)
	}
	sums := []float32{}
	prods := []float32{}
	for range freqs {
		sums = append(sums, 1)
		prods = append(prods, 1)
	}
	rand.Seed(uint64(time.Now().UnixNano()))
	c := 0
	var index int
	success := false
	for !success {
		index, success, err = step(words, model, freqs, sums, prods)
		freqs = append(freqs[:index], freqs[index+1:]...)
		sums = append(sums[:index], sums[index+1:]...)
		prods = append(prods[:index], prods[index+1:]...)
		words = append(words[:index], words[index+1:]...)
		if err != nil {
			log.Println(err)
		}
		c += 1
	}
	fmt.Println("done", c)
}
