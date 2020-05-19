/**
 * FCRLM proposed in : "Retrieving Potential Causes from a Query Event" --- SIGIR 2020
 * Factored Causal Model (on top of Relevance Feedback Model RM3)
 * References:
 *      1. Relevance Based Language Model - Victor Lavrenko - SIGIR-2001
 *      2. UMass at TREC 2004: Novelty and HARD - Nasreen Abdul-Jaleel - TREC-2004
 */

package FCRLM;

import static common.CommonVariables.FIELD_BOW;
import static common.CommonVariables.FIELD_FULL_BOW;
import static common.CommonVariables.FIELD_ID;
import common.EnglishAnalyzerWithSmartStopword;
import common.TRECQuery;
import common.TRECQueryParser;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopScoreDocCollector;
import org.apache.lucene.search.similarities.AfterEffectB;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.BasicModelIF;
import org.apache.lucene.search.similarities.DFRSimilarity;
import org.apache.lucene.search.similarities.DefaultSimilarity;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.search.similarities.LMJelinekMercerSimilarity;
import org.apache.lucene.search.similarities.NormalizationH2;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 *
 * @author Dwaipayan (implemented for RM3 Model)
 * Modified by Suchana (implemented for FCRLM)
 */

public class RelevanceBasedCausalModel {

    Properties      prop;
    String          indexPath;
    String          queryPath;               // path of the query file
    File            queryFile;               // the query file
    String          stopFilePath;
    IndexReader     indexReader;
    IndexSearcher   indexSearcher;
    String          resPath;                 // path of the res file
    FileWriter      resFileWriter;           // the res file writer
    FileWriter      baselineFileWriter;      // the res file writer
    int             numHits;                 // number of document to retrieveWithExpansionTermsFromFile
    String          runName;                 // name of the run
    List<TRECQuery> queries;
    File            indexFile;               // place where the index is stored
    Analyzer        analyzer;                // the analyzer
    boolean         boolIndexExists;         // boolean flag to indicate whether the index exists or not
    String          fieldToSearch;           // the field in the index to be searched
    String          fieldForFeedback;        // field, to be used for feedback
    TRECQueryParser trecQueryparser;
    int             simFuncChoice;
    float           param1, param2;
    long            vocSize;                 // vocabulary size
    //RM3             rm3;                   // reference to RM3 to use Relevance Feedback Model
    FactoredRLM     frlm;                    // reference to factored causal model     
    HashMap<String, TopDocs> allTopDocsFromFileHashMap;     // For feedback from file, to contain all topdocs from file
    float           mixingLambda;            // mixing weight, used for doc-col weight distribution
    int             numFeedbackTermsTopical; // number of feedback terms at the first step
    int             numFeedbackTermsCausal;  // number of feedback terms at the second step
    int             numFeedbackDocs;         // number of feedback documents
    float           QMIX;                    // query mix to weight between P(w|R) and P(w|Q)
    

    public RelevanceBasedCausalModel(Properties prop) throws IOException, Exception {

        this.prop = prop;
        /* property file loaded */

        /* setting the analyzer with English Analyzer with Smart stopword list */
        EnglishAnalyzerWithSmartStopword engAnalyzer;
        stopFilePath = prop.getProperty("stopFilePath");
        if (null == stopFilePath)
            engAnalyzer = new common.EnglishAnalyzerWithSmartStopword();
        else
            engAnalyzer = new common.EnglishAnalyzerWithSmartStopword(stopFilePath);
        analyzer = engAnalyzer.setAndGetEnglishAnalyzerWithSmartStopword();
        /* analyzer set: analyzer */

        /* index path setting */
        indexPath = prop.getProperty("indexPath");
        System.out.println("indexPath set to: " + indexPath);
        indexFile = new File(prop.getProperty("indexPath"));
        Directory indexDir = FSDirectory.open(indexFile.toPath());

        if (!DirectoryReader.indexExists(indexDir)) {
            System.err.println("Index doesn't exists in "+indexPath);
            boolIndexExists = false;
            System.exit(1);
        }
        fieldToSearch = prop.getProperty("fieldToSearch", FIELD_FULL_BOW);
        fieldForFeedback = prop.getProperty("fieldForFeedback", FIELD_BOW);
        //System.out.println("Searching field for retrieval: " + fieldToSearch);
        //System.out.println("Field for Feedback: " + fieldForFeedback);
        /* index path set */

        simFuncChoice = Integer.parseInt(prop.getProperty("similarityFunction"));
        if (null != prop.getProperty("param1"))
            param1 = Float.parseFloat(prop.getProperty("param1"));
        if (null != prop.getProperty("param2"))
            param2 = Float.parseFloat(prop.getProperty("param2"));

        /* setting indexReader and indexSearcher */
        indexReader = DirectoryReader.open(FSDirectory.open(indexFile.toPath()));
        indexSearcher = new IndexSearcher(indexReader);
        setSimilarityFunction(simFuncChoice, param1, param2);
        /* indexReader and searcher set */

        /* setting query path */
        queryPath = prop.getProperty("queryPath");
        System.out.println("queryPath set to: " + queryPath);
        queryFile = new File(queryPath);
        /* query path set */

        /* constructing the query */
        trecQueryparser = new TRECQueryParser(queryPath, analyzer, fieldToSearch);
        queries = constructQueries();
        /* constructed the query */

        /* numFeedbackTerms = number of top terms to select in two steps */
        numFeedbackTermsTopical = Integer.parseInt(prop.getProperty("numFeedbackTermsTopical"));
        numFeedbackTermsCausal = Integer.parseInt(prop.getProperty("numFeedbackTermsCausal"));
        /* numFeedbackDocs = number of top documents to select */
        numFeedbackDocs = Integer.parseInt(prop.getProperty("numFeedbackDocs"));

        /* setting mixing Lambda */
        if(param1>0.99)
            mixingLambda = 0.8f;
        else
            mixingLambda = param1;

        numHits = Integer.parseInt(prop.getProperty("numHits","1000"));
        QMIX = Float.parseFloat(prop.getProperty("rm3.queryMix"));
        
        frlm = new FactoredRLM(this);

        /* setting res path */
        setRunName_ResFileName();
        resFileWriter = new FileWriter(resPath);
        System.out.println("Result will be stored in: "+resPath);
        /* res path set */
    }
    

    /**
     * Sets indexSearcher.setSimilarity() with parameter(s)
     * @param choice similarity function selection flag
     * @param param1 similarity function parameter 1
     * @param param2 similarity function parameter 2
     */
    private void setSimilarityFunction(int choice, float param1, float param2) {

            switch(choice) {
            case 0:
                indexSearcher.setSimilarity(new DefaultSimilarity());
                System.out.println("Similarity function set to DefaultSimilarity");
                break;
            case 1:
                indexSearcher.setSimilarity(new BM25Similarity(param1, param2));
                System.out.println("Similarity function set to BM25Similarity"
                    + " with parameters: " + param1 + " " + param2);
                break;
            case 2:
                indexSearcher.setSimilarity(new LMJelinekMercerSimilarity(param1));
                System.out.println("Similarity function set to LMJelinekMercerSimilarity"
                    + " with parameter: " + param1);
                break;
            case 3:
                indexSearcher.setSimilarity(new LMDirichletSimilarity(param1));
                System.out.println("Similarity function set to LMDirichletSimilarity"
                    + " with parameter: " + param1);
                break;
            case 4:
                indexSearcher.setSimilarity(new DFRSimilarity(new BasicModelIF(), new AfterEffectB(), new NormalizationH2()));
                System.out.println("Similarity function set to DFRSimilarity with default parameters");
                break;
        }
    } // ends setSimilarityFunction()
    

    /**
     * Sets runName and resPath variables depending on similarity functions.
     */
    private void setRunName_ResFileName() {

        Similarity s = indexSearcher.getSimilarity(true);
        runName = s.toString()+"-D"+numFeedbackDocs+"-T"+numFeedbackTermsTopical+"-C"+numFeedbackTermsCausal;
        runName += "-queryMix-"+QMIX;
        runName += "-" + fieldToSearch + "-" + fieldForFeedback;
        runName = runName.replace(" ", "").replace("(", "").replace(")", "").replace("00000", "");
        if(null == prop.getProperty("resPath"))
            resPath = "/home/suchana/";
        else
            resPath = prop.getProperty("resPath");
        resPath = resPath+queryFile.getName()+"-"+runName + ".res";
    } // ends setRunName_ResFileName()

    /**
     * Parses the query from the file and makes a List<TRECQuery> 
     *  containing all the queries (RAW query read)
     * @return A list with the all the queries
     * @throws Exception 
     */
    private List<TRECQuery> constructQueries() throws Exception {

        trecQueryparser.queryFileParse();
        return trecQueryparser.queries;
    } // ends constructQueries()
    

    public void retrieveAll() throws Exception {

        ScoreDoc[] hits;
        TopDocs topDocsPRD1, topDocsPRD2, topDocsFinal;
        TopScoreDocCollector collector;
        HashMap<String, WordProbability> hashmap_PwGivenR, hashmap_PwGivenR_causal;;

        for (TRECQuery query : queries) {
            collector = TopScoreDocCollector.create(numHits);
            Query luceneQuery = trecQueryparser.getAnalyzedQuery(query);

            System.out.println("\n" + query.qid +": Initial query: " + luceneQuery.toString(fieldToSearch));

            /* PRF - initial retrieval performed */
            indexSearcher.search(luceneQuery, collector);
            topDocsPRD1 = collector.topDocs();
            //System.out.println("docs retrieved : " + topDocsPRD1.totalHits);
            /* PRF */

            StringBuffer resBuffer;
            
            frlm.setFeedbackStats(topDocsPRD1, luceneQuery.toString(fieldToSearch).split(" "), this);
            
            /**
             * HashMap of P(w|R) for 'numFeedbackTerms' terms with top P(w|R) among each w in R,
             * keyed by the term with P(w|R) as the value.
             * T1 = normalized RM1(D1)--<sorted> 
             * T1'= normalized top n terms of T1--<with highest weights>
             * EQ1 = RM3(T1',Q,alpha) and retrieve
             */

            hashmap_PwGivenR = frlm.RM3(query, topDocsPRD1, luceneQuery.toString(fieldToSearch).split(" "));
            /* EQ1 = RM3(T1',Q, alpha) */
            
            BooleanQuery booleanQuery;

            booleanQuery = frlm.getExpandedQuery(hashmap_PwGivenR, query);
            System.out.println("\nRe-retrieval after 1st level estimation with EQ1 :");
            System.out.println(booleanQuery.toString(fieldToSearch));
            collector = TopScoreDocCollector.create(numHits);
            indexSearcher.search(booleanQuery, collector);      //retrieve with EQ1
            
            /* D2 = top k docs of search (EQ1,C) */
            topDocsPRD2 = collector.topDocs(); 
            hits = topDocsPRD2.scoreDocs;
                if(hits == null)
                System.out.println("Nothing found");

            int hits_length = hits.length;

            frlm.setFeedbackStats(topDocsPRD2, booleanQuery.toString(fieldToSearch).split(" "), this);
            
            /**
             * HashMap of P(w|R) for 'numFeedbackTerms' terms with top P(w|R) among each w in R,
             * keyed by the term with P(w|R) as the value.
             * 
             * T2 = normalized RM1(D2)---<sorted>
             * T2'= normalized top n terms of T2 that are overlapping with T1'
             * T2"= {t2 / t1} ; t1=term from T1' & t2=term from T2'
             * EQ2 = RM3(T2",Q,alpha) and retrieve
             **/
            
            hashmap_PwGivenR_causal = frlm.RM3_overloaded(booleanQuery.toString(fieldToSearch).split(" "), topDocsPRD2, 
                    luceneQuery.toString(fieldToSearch).split(" "), hashmap_PwGivenR);
            
            BooleanQuery booleanQuery_causal;

            booleanQuery_causal = frlm.getExpandedQuery_Overloaded(hashmap_PwGivenR_causal, booleanQuery.toString(fieldToSearch).split(" "));
            System.out.println("Final-retrieval after causal estimation with EQ2 :");
            System.out.println(booleanQuery_causal.toString(fieldToSearch));
            collector = TopScoreDocCollector.create(numHits);
            indexSearcher.search(booleanQuery_causal, collector);

            topDocsFinal = collector.topDocs();
            hits = topDocsFinal.scoreDocs;
                if(hits == null)
                System.out.println("Nothing found");

            int hits_length_level2 = hits.length;

            resFileWriter = new FileWriter(resPath, true);

            /* res file in TREC format with doc text (7 columns) */
            resBuffer = new StringBuffer();
            for (int i = 0; i < hits_length_level2; ++i) {
                int docId = hits[i].doc;
                Document d = indexSearcher.doc(docId);
                resBuffer.append(query.qid).append("\tQ0\t").
                append(d.get(FIELD_ID)).append("\t").
                append((i)).append("\t").
                append(hits[i].score).append("\t").
                append(runName).append("\t").
                append(d.get(FIELD_BOW)).append("\n");
            }
            resFileWriter.write(resBuffer.toString());
            resFileWriter.close();
        } // ends for each query
    } // ends retrieveAll
    

    public static void main(String[] args) throws IOException, Exception {

        String usage = "java RelevanceBasedLanguageModel <properties-file>\n"
                + "Properties file must contain the following fields:\n"
                + "1. stopFilePath: path of the stopword file\n"
                + "2. indexPath: Path of the index\n"
                + "3. queryPath: path of the query file (in proper xml format)\n"
                + "4. resPath: path of the directory to store res file\n"
                + "5. numFeedbackDocs: number of feedback documents to use\n"
                + "6. numFeedbackTermsTopical: number of feedback terms to use at 1st level"
                + "7. numFeedbackTermsCausal: number of feedback terms to use at 2nd level"
                + "8. rm3.queryMix (0.0-1.0): query mix to weight between P(w|R) and P(w|Q)\n"
                + "9. similarityFunction: 0.DefaultSimilarity, 1.BM25Similarity, 2.LMJelinekMercerSimilarity, 3.LMDirichletSimilarity\n";               
                
        Properties prop = new Properties();

        if(1 != args.length) {
            System.out.println("Usage: " + usage);
            args = new String[1];
            args[0] = "fcrlm-0.4-query_test.xml.D-10.topical-10.causal-10.properties";
            System.exit(1);
        }
        prop.load(new FileReader(args[0]));
        RelevanceBasedCausalModel rbcm = new RelevanceBasedCausalModel(prop);

        rbcm.retrieveAll();
    } // ends main()
}
