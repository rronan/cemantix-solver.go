package server

import (
	"fmt"
	"log"
	"net/http"
	"os"

	"code.sajari.com/word2vec"
)

const WORD2VEC_PATH = "frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"

var WORD = os.Args[len(os.Args)-1]
var MODEL *word2vec.Model

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

func hello(w http.ResponseWriter, req *http.Request) {
	word := "bonjour"
	res, err := MODEL.Cos(word2vec.Expr{WORD: 1}, word2vec.Expr{word: 1})

	fmt.Fprintf(w, "hello\n")
}

func main() {
	MODEL = loadBinary(WORD2VEC_PATH)
	http.HandleFunc("/hello", hello)
	http.ListenAndServe(":8090", nil)
}
