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
const CEMANTIX_URL = "https://cemantix.certitudes.org/score"

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
	r, err := http.NewRequest("POST", CEMANTIX_URL, bytes.NewBuffer(body))
	if err != nil {
		return 0, err
	}
	r.Header.Add("Content-Type", "application/x-www-form-urlencoded")
	r.Header.Add("Origin", "https://cemantix.certitudes.org")
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

func convertTo64(ar []float32) []float64 {
	newar := make([]float64, len(ar))
	var v float32
	var i int
	for i, v = range ar {
		newar[i] = float64(v)
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

func handlePanic(weights []float32, weights64 []float64) {
	r := recover()
	if r != nil {
		fmt.Println(weights64[:100])
		panic(r)
	}
}

func step(weights []float32, words []string, model *word2vec.Model) (bool, error) {
	weights64 := convertTo64(weights)
	sampler := sampleuv.NewWeighted(weights64, nil)
	defer handlePanic(weights, weights64)
	index, _ := sampler.Take()
	word := words[index]
	weights = append(weights[:index], weights[index+1:]...)
	words = append(words[:index], words[index+1:]...)
	score, err := getScore(word)
	fmt.Println(word, score)
	if err != nil {
		return false, err
	}
	if score == 1 {
		return true, nil
	}
	similarities, err := computeSimilarities(word, words, model)
	if err != nil {
		return false, err
	}
	sum := float32(0.0)
	for i, sim := range similarities {
		weights[i] /= math32.Abs(score - sim)
		sum += weights[i]
	}
	for i := range similarities {
		if sum == 0 {
			weights[i] = 1.0 / float32(len(weights))
		} else {
			weights[i] /= sum
		}
	}
	return false, nil
}

func main() {
	model := loadBinary(WORD2VEC_PATH)
	records := readCsvFile(LEXICON_PATH)
	words, weights, err := processCSV(records, model)
	if err != nil {
		log.Fatal(err)
	}
	rand.Seed(uint64(time.Now().UnixNano()))
	c := 0
	success := false
	for !success {
		success, err = step(weights, words, model)
		if err != nil {
			log.Println(err)
		}
		c += 1
	}
	fmt.Println("done", c)
}
