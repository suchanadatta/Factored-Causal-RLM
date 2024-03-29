# FCRLM : In search of chain of causes behind Query Events (SIGIR-2020) 
-- Suchana Datta, Debasis Ganguly, Dwaipayan Roy, Francesca Bonin, Charles Jochim and Mandar Mitra

FCRLM (Factored Causal ReLevance Model) is written on top of Relevance Feedback Model implemented using java by Dwaipayan Roy (https://github.com/dwaipayanroy/Relevance-Model)

![Alt text](fcrlm_model.png?raw=true "Title")

> Build using Ant:
`````
ant
`````

> Set parameter in 'run.sh' file:
`````````````````````````````````````````````
stopFilePath="The path of the stopword file" 
## ./src/resources/smart-stopwords
`````````````````````````````````````````````

> Run the script file followed by 8 parameters:
``````````````````````````````````````````````````````````````````````````````````````````
./scripts/run.sh
1. The path of the Lucene index on which the retrieval will be performed.
2. The path of the query file in complete XML format.
3. Directory path in which the .res file will be saved in TREC 6-column .res format.
4. Number of pseudo-relevant documents.
5. Number of feedback terms from pseudo-topical relevant set.
6. Number of feedback terms from pseudo-causal relevant set.
7. RM3-QueryMix (0.0-1.0): to weight between P(w|R) and P(w|Q).
8. SimilarityFunction: 0.DefaultSimilarity, 1.BM25Similarity, 2.LMJelinekMercerSimilarity, 3.LMDirichletSimilarity.
``````````````````````````````````````````````````````````````````````````````````````````

If you are using this model, please consider citing our work : 
``````````````````````````````````````````````````````````````
@inproceedings{DBLP:conf/sigir/DattaGRBJM20,
  author    = {Suchana Datta and
               Debasis Ganguly and
               Dwaipayan Roy and
               Francesca Bonin and
               Charles Jochim and
               Mandar Mitra},
  title     = {Retrieving Potential Causes from a Query Event},
  booktitle = {{SIGIR}},
  pages     = {1689--1692},
  publisher = {{ACM}},
  year      = {2020}
}
``````````````````````````````````````````````````````````````
