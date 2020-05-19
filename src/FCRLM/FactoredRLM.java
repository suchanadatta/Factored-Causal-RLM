package FCRLM;

import common.DocumentVector;
import common.PerTermStat;
import common.TRECQuery;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;

/**
 *
 * @author Dwaipayan (implemented for RM3 Model)
 * Modified by Suchana (implemented for FCRLM)
 */


public class FactoredRLM {

    IndexReader     indexReader;
    IndexSearcher   indexSearcher;
    String          fieldForFeedback;          // the field of the index which will be used for feedback
    Analyzer        analyzer;    
    int             numFeedbackTermsTopical;   // number of feedback terms
    int             numFeedbackTermsCausal;
    int             numFeedbackDocs;           // number of feedback documents
    float           mixingLambda;              // mixing weight, used for doc-col weight adjustment
    float           QMIX;                      // query mixing parameter; to be used for RM3
    float           linearCombo;

    RelevanceBasedCausalModel rbcm;   
    /**
     * Hashmap of Vectors of all feedback documents, keyed by luceneDocId.
     */
    HashMap<Integer, DocumentVector>    feedbackDocumentVectors;
    /**
     * HashMap of PerTermStat of all feedback terms, keyed by the term.
     */
    HashMap<String, PerTermStat>        feedbackTermStats;
    /**
     * HashMap of P(Q|D) for all feedback documents, keyed by luceneDocId.
     */
    HashMap<Integer, Float> hash_P_Q_Given_D;

    TopDocs         topDocs;
    long            vocSize;                    // vocabulary size
    long            docCount;                   // number of documents in the collection

    /**
     * List, for sorting the words in non-increasing order of probability.
     */
    List<WordProbability> list_PwGivenR;
    /**
     * HashMap of P(w|R) for 'numFeedbackTerms' terms with top P(w|R) among each w in R,
     * keyed by the term with P(w|R) as the value.
     */
    HashMap<String, WordProbability> hashmap_PwGivenR;
    HashMap<String, WordProbability> hashmap_PwGivenR_final;
    

    public FactoredRLM(RelevanceBasedCausalModel rbcm) throws IOException {

        this.rbcm = rbcm;
        this.indexReader = rbcm.indexReader;
        this.indexSearcher = rbcm.indexSearcher;
        this.analyzer = rbcm.analyzer;
        this.fieldForFeedback = rbcm.fieldForFeedback;
        this.numFeedbackDocs = rbcm.numFeedbackDocs;
        this.numFeedbackTermsTopical = rbcm.numFeedbackTermsTopical;
        this.numFeedbackTermsCausal = rbcm.numFeedbackTermsCausal;
        this.mixingLambda = rbcm.mixingLambda;
        this.QMIX = rbcm.QMIX;
        vocSize = getVocabularySize();
        docCount = indexReader.maxDoc();        // total number of documents in the index

    }
    
    
    /**
     * Returns the vocabulary size of the collection for the field 'fieldForFeedback'.
     * @return vocSize : Total number of terms in the vocabulary
     * @throws IOException IOException
     */
    private long getVocabularySize() throws IOException {

        Fields fields = MultiFields.getFields(indexReader);
        Terms terms = fields.terms(fieldForFeedback);
        if(null == terms) {
            System.err.println("Field: "+fieldForFeedback);
            System.err.println("Error buildCollectionStat(): terms Null found");
        }
        vocSize = terms.getSumTotalTermFreq();  // total number of terms in the index in that field

        return vocSize;                         // total number of terms in the index in that field
    }
    

    /**
     * Sets the following variables with feedback statistics: to be used consequently.<p>
     * {@link #feedbackDocumentVectors},<p> 
     * {@link #feedbackTermStats}, <p>
     * {@link #hash_P_Q_Given_D}
     * @param topDocs
     * @param analyzedQuery
     * @param rbcm
     * @throws IOException 
     */
    public void setFeedbackStats(TopDocs topDocs, String[] analyzedQuery, RelevanceBasedCausalModel rbcm) throws IOException {

        feedbackDocumentVectors = new HashMap<>();
        feedbackTermStats = new HashMap<>();
        hash_P_Q_Given_D = new HashMap<>();

        ScoreDoc[] hits;
        int hits_length;
        hits = topDocs.scoreDocs;
        hits_length = hits.length;                                      // number of documents retrieved in the first retrieval
        
        for (int i = 0; i < Math.min(numFeedbackDocs, hits_length); i++) {
            // for each feedback document
            int luceneDocId = hits[i].doc;
            Document d = indexSearcher.doc(luceneDocId);
            DocumentVector docV = new DocumentVector(rbcm.fieldForFeedback);
            docV = docV.getDocumentVector(luceneDocId, indexReader);
            if(docV == null)
                continue;
            feedbackDocumentVectors.put(luceneDocId, docV);             // the feedback document vector is added in the list

            for (Map.Entry<String, PerTermStat> entrySet : docV.docPerTermStat.entrySet()) {
            // for each term of that feedback document
                String key = entrySet.getKey();
                PerTermStat value = entrySet.getValue();
                
                if(null == feedbackTermStats.get(key)) {
                // this feedback term is not already put in the hashmap, hence to be added;
                    Term termInstance = new Term(fieldForFeedback, key);
                    long cf = indexReader.totalTermFreq(termInstance);  // CF: Returns the total number of occurrences of term across all documents (the sum of the freq() for each doc that has this term).
                    long df = indexReader.docFreq(termInstance);        // DF: Returns the number of documents containing the term

                    feedbackTermStats.put(key, new PerTermStat(key, cf, df));
                }
            } // ends for each term of that feedback document
        } // ends for each feedback document

        // Calculating P(Q|d) for each feedback documents
        for (Map.Entry<Integer, DocumentVector> entrySet : feedbackDocumentVectors.entrySet()) {
            // for each feedback document
            int luceneDocId = entrySet.getKey();
            DocumentVector docV = entrySet.getValue();

            float p_Q_GivenD = 0;
            float smoothMLE = 0;
            
            for (String qTerm : analyzedQuery){
                if(qTerm.contains("^")){
                    String[] expQuery = qTerm.split("\\^");
                    qTerm = expQuery[0];
                }
                smoothMLE = return_Smoothed_MLE_Log(qTerm, docV);
                p_Q_GivenD += smoothMLE;
            }                
            if(null == hash_P_Q_Given_D.get(luceneDocId)){
                hash_P_Q_Given_D.put(luceneDocId, p_Q_GivenD);
            }
            else {
                System.err.println("Error while pre-calculating P(Q|d). "
                + "For luceneDocId: " + luceneDocId + ", P(Q|d) already existed.");
            }
        }
    }

    
    public float return_Smoothed_MLE_Log(String t, DocumentVector dv) throws IOException {
        
        float smoothedMLEofTerm = 1;
        PerTermStat docPTS;
        docPTS = dv.docPerTermStat.get(t);
        PerTermStat colPTS = feedbackTermStats.get(t);

        if (colPTS != null) 
            smoothedMLEofTerm = 
                ((docPTS!=null)?(mixingLambda * (float)docPTS.getCF() / (float)dv.getDocSize()):(0)) /
                ((feedbackTermStats.get(t)!=null)?((1.0f-mixingLambda)*(float)feedbackTermStats.get(t).getCF()/(float)vocSize):0);
     
        return (float)Math.log(1+smoothedMLEofTerm);

    } // ends return_Smoothed_MLE_Log()

    
    public float getCollectionProbability(String term, IndexReader reader, String fieldName) throws IOException {

        Term termInstance = new Term(fieldName, term);
        long termFreq = reader.totalTermFreq(termInstance); // CF: Returns the total number of occurrences of term across all documents (the sum of the freq() for each doc that has this term).

        return (float) termFreq / (float) vocSize;
    }
    

    /**
     * Returns MLE of a query term q in Q;<p>
     * P(w|Q) = tf(w,Q)/|Q|
     * @param qTerms all query terms
     * @param qTerm query term under consideration
     * @return MLE of qTerm in the query qTerms
     */
    public float returnMLE_of_q_in_Q(String[] qTerms, String qTerm) {

        int count=0;
        for (String queryTerm : qTerms)
            if (qTerm.equals(queryTerm))
                count++;
        return ( (float)count / (float)qTerms.length );
    } // ends returnMLE_of_w_in_Q()

    
    /**
     * RM1: IID Sampling <p>
     * Returns 'hashmap_PwGivenR' containing all terms of PR docs (PRD) with 
     * weights calculated using IID Sampling <p>
     * P(w|R) = \sum{d\in PRD} {smoothedMLE(w,d)*smoothedMLE(Q,d)}
     * Reference: Relevance Based Language Model - Victor Lavrenko (SIGIR-2001)
     * @param query The query
     * @param topDocs Initial retrieved document list
     * @return 'hashmap_PwGivenR' containing all terms of PR docs with weights
     * @throws Exception 
     */
    public HashMap RM1(TRECQuery query, TopDocs topDocs) throws Exception {

        float p_W_GivenR_one_doc;
        list_PwGivenR = new ArrayList<>();
        hashmap_PwGivenR = new LinkedHashMap<>();
        int expansionTermCount = 0;
        float normFactor = 0;
        
        /* Calculating for each w_i in R: P(w_i|R)~P(wi, q1 ... qk)
           P(wi, q1 ... qk) = \sum{d\in PRD} {P(w|D)*\prod_{i=1... k} {P(qi|D}} */

        for (Map.Entry<String, PerTermStat> entrySet : feedbackTermStats.entrySet()) {
            // for each t in R:
            String t = entrySet.getKey();
            p_W_GivenR_one_doc = 0;

            for (Map.Entry<Integer, DocumentVector> docEntrySet : feedbackDocumentVectors.entrySet()) {
            // for each doc in RF-set
                int luceneDocId = docEntrySet.getKey();
                p_W_GivenR_one_doc += return_Smoothed_MLE_Log(t, feedbackDocumentVectors.get(luceneDocId)) *
                    hash_P_Q_Given_D.get(luceneDocId);
            }
            list_PwGivenR.add(new WordProbability(t, p_W_GivenR_one_doc));
        }
        
        /* sorting list in descending order
           T1 = Normalized RM1(D1) -- <sorted> */
        Collections.sort(list_PwGivenR, new Comparator<WordProbability>(){
            @Override
            public int compare(WordProbability t, WordProbability t1) {
                return t.p_w_given_R<t1.p_w_given_R?1:t.p_w_given_R==t1.p_w_given_R?0:-1;
            }}); 
        // -- sorted list in descending order
        
        // T1'= normalized top n terms of t1 <with highest weights>
        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
                expansionTermCount++;
                normFactor += singleTerm.p_w_given_R;
                if(expansionTermCount>=numFeedbackTermsTopical)
                    break;
            }
            //* else: The t is already entered in the hash-map 
        } //hashmap_PwGivenR has all terms of PRDs along with their probabilities 

        /* selecting top numFeedbackTerms terms and normalize */
        // ++ Normalizing 
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            WordProbability wp = entrySet.getValue();
            wp.p_w_given_R /= normFactor;
            wp.expansionWeight = wp.p_w_given_R;
        }
        // -- Normalizing done
        
        //System.out.println("No of terms from RM1 : " + hashmap_PwGivenR.size());       
        return hashmap_PwGivenR; 
    }   // ends RM1()
    
    
    public HashMap RM1_overloaded(String[] analyzedQuery , TopDocs topDocs) throws Exception {

        float p_W_GivenR_one_doc;
        list_PwGivenR = new ArrayList<>();
        hashmap_PwGivenR = new LinkedHashMap<>();
        int expansionTermCount = 0;
        float normFactor = 0;
        
        /* Calculating for each wi in R: P(wi|R)~P(wi, q1 ... qk)
           P(wi, q1 ... qk) = \sum{d\in PRD} {P(w|D)*\prod_{i=1... k} {P(qi|D}} */

        for (Map.Entry<String, PerTermStat> entrySet : feedbackTermStats.entrySet()) {
            // for each t in R:
            String t = entrySet.getKey();
            p_W_GivenR_one_doc = 0;

            for (Map.Entry<Integer, DocumentVector> docEntrySet : feedbackDocumentVectors.entrySet()) {
            // for each doc in RF-set
                int luceneDocId = docEntrySet.getKey();
                p_W_GivenR_one_doc += 
                    return_Smoothed_MLE_Log(t, feedbackDocumentVectors.get(luceneDocId)) *
                    hash_P_Q_Given_D.get(luceneDocId);
            }
            list_PwGivenR.add(new WordProbability(t, p_W_GivenR_one_doc));
        }
        
        /* sorting list in descending order
           T2 = Nomalized RM1(D2)--sorted */
        Collections.sort(list_PwGivenR, new Comparator<WordProbability>(){
            @Override
            public int compare(WordProbability t, WordProbability t1) {
                return t.p_w_given_R<t1.p_w_given_R?1:t.p_w_given_R==t1.p_w_given_R?0:-1;
            }}); 
        // -- sorted list in descending order
        
        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
                expansionTermCount++;
                normFactor += singleTerm.p_w_given_R;
                if(expansionTermCount>=numFeedbackTermsCausal)
                    break;
            }
            //* else: The t is already entered in the hash-map 
        } //hashmap_PwGivenR has all terms of PRDs along with their probabilities 

        /* selecting top numFeedbackTerms terms and normalize */
        // ++ Normalizing 
        //System.out.println("NORM FACTOR : "+ normFactor);
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            WordProbability wp = entrySet.getValue();
            wp.p_w_given_R /= normFactor;
            wp.expansionWeight = wp.p_w_given_R;
            //System.out.println("Candidate term : "+entrySet.getKey()+"\tExpansion weight : "+wp.expansionWeight);
        }
        // -- Normalizing done
        //System.out.println("No. of terms from RM1_overloaded : " + hashmap_PwGivenR.size());
        return hashmap_PwGivenR;
    }   // ends RM1_overloaded()
    
    
    /**
     * RM3 <p>
     * P(w|R) = QueryMix*RM1 + (1-QueryMix)*P(w|Q) <p>
     * Reference: Nasreen Abdul Jaleel - TREC 2004 UMass Report <p>
     * @param query The query 
     * @param topDocs Initially retrieved document list
     * @return hashmap_PwGivenR: containing numFeedbackTerms expansion terms with normalized weights
     * @throws Exception 
     */
    public HashMap RM3(TRECQuery query, TopDocs topDocs, String[] initialQuery) throws Exception {

        hashmap_PwGivenR = new LinkedHashMap<>();
        float normFactor = 0;
        // hashmap_PwGivenR has all terms of PRDs along with their probabilities
        hashmap_PwGivenR = RM1(query, topDocs);

        /* Each w of R: P(w|R) to be (1-QMIX)*P(w|R) */
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            String key = entrySet.getKey();
            WordProbability value = entrySet.getValue();
            value.p_w_given_R = value.p_w_given_R * (1.0f-QMIX);
            normFactor += value.p_w_given_R;
        }

        /* Now P(w|R) = (1-QMIX)*P(w|R)
           Each w which are also query terms: P(w|R) += QMIX*P(w|Q)
           P(w|Q) = tf(w,Q)/|Q| */
        
        for (String qTerm : initialQuery) {
            WordProbability oldProba = hashmap_PwGivenR.get(qTerm);
            float newProb = QMIX * returnMLE_of_q_in_Q(initialQuery, qTerm);
            normFactor += newProb;
            if (null != oldProba) { // qTerm is in R
                oldProba.p_w_given_R += newProb;
                hashmap_PwGivenR.put(qTerm, oldProba);
            }
            else  // the qTerm is not in R
                hashmap_PwGivenR.put(qTerm, new WordProbability(qTerm, newProb));
        }

        // ++ Normalizing
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            WordProbability wp = entrySet.getValue();
            wp.p_w_given_R /= normFactor;
            wp.expansionWeight = wp.p_w_given_R;
        }
        // -- Normalizing done
        
        //System.out.println("No. of terms from RM3 : " + hashmap_PwGivenR.size());
        return hashmap_PwGivenR;
    } // end RM3()
    
    
    public HashMap RM3_overloaded(String[] analyzedQuery , TopDocs topDocs, String[] initialQuery, 
            HashMap<String, WordProbability> hashMap_PwGivenR_RM3) throws Exception {
        
        hashmap_PwGivenR = new LinkedHashMap<>();
        float normFactor = 0, epsilon = 0, p_w_given_Rc_final = 0;
        int expansionTermCount = 0;
        hashmap_PwGivenR = RM1_overloaded(analyzedQuery, topDocs);
        // hashmap_PwGivenR has all terms of PRDs along with their probabilities 

        for(Map.Entry<String, WordProbability> termScore_2nd : hashmap_PwGivenR.entrySet()){
            normFactor += termScore_2nd.getValue().p_w_given_R;
        }
        epsilon = normFactor / hashmap_PwGivenR.size();
        // System.out.println("The value of EPSILON : " + epsilon);
        
        /* T2'= normalized top n terms of T2 that are overlapping with T1'
           T2"= {t2 / t1} ; t1=term from T1' & t2=term from T2' */
        
        normFactor = 0;
        HashMap<String, WordProbability> hashmap_PwGivenR_final = new LinkedHashMap<>();
        for(Map.Entry<String, WordProbability> entrySet_level2 : hashmap_PwGivenR.entrySet()){
            for(Map.Entry<String, WordProbability> entrySet_level1 : hashMap_PwGivenR_RM3.entrySet()){
                if(entrySet_level2.getKey().equalsIgnoreCase(entrySet_level1.getKey())){
                    p_w_given_Rc_final = entrySet_level2.getValue().p_w_given_R / entrySet_level1.getValue().p_w_given_R;
                    normFactor += p_w_given_Rc_final;
                    hashmap_PwGivenR_final.put(entrySet_level2.getKey(), new WordProbability(entrySet_level2.getKey(), p_w_given_Rc_final));
                    //System.out.println("term : "+ entrySet_level2.getKey() + "\t proba : "+ p_w_given_Rc_final);
                }
                else{
                    p_w_given_Rc_final = entrySet_level2.getValue().p_w_given_R / epsilon;
                    hashmap_PwGivenR_final.put(entrySet_level2.getKey(), new WordProbability(entrySet_level2.getKey(), p_w_given_Rc_final));
                    normFactor += p_w_given_Rc_final;
                }
            }
        }
        
        // ++ Normalizing 
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR_final.entrySet()) {
            WordProbability wp = entrySet.getValue();
            wp.p_w_given_R /= normFactor;
        }
        // -- Normalizing done

        normFactor = 0;
        /* Each w of R: P(w|R) to be (1-QMIX)*P(w|R) */
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            String key = entrySet.getKey();
            WordProbability value = entrySet.getValue();
            value.p_w_given_R = value.p_w_given_R * (1.0f-QMIX);
            normFactor += value.p_w_given_R;
        }

        // Now P(w|R) = (1-QMIX)*P(w|R)
        //* Each w which are also query terms: P(w|R) += QMIX*P(w|Q)
        //      P(w|Q) = tf(w,Q)/|Q|
        
        for (String qTerm : initialQuery) {
            WordProbability oldProba = hashmap_PwGivenR_final.get(qTerm);
            //System.out.println("OLD PROBABILITY : "+ oldProba.p_w_given_R);
            float newProb = QMIX * returnMLE_of_q_in_Q(initialQuery, qTerm);
            //System.out.println("NEW PROBABILITY : "+ newProb);
            normFactor += newProb;
            if (null != oldProba) { // qTerm is in R
                oldProba.p_w_given_R += newProb;
                hashmap_PwGivenR_final.put(qTerm, oldProba);
            }
            else{  // the qTerm is not in R
                hashmap_PwGivenR_final.put(qTerm, new WordProbability(qTerm, newProb));
            }
        }
        
        // +++ Normalizing
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR_final.entrySet()) {
            int flag = 0;
            for(int i=0; i<initialQuery.length; i++){
                if(entrySet.getKey().equalsIgnoreCase(initialQuery[i])){
                    WordProbability wp = entrySet.getValue();
                    wp.expansionWeight = wp.p_w_given_R;
                    hashmap_PwGivenR_final.put(entrySet.getKey(), wp);
                    flag = 1;
                    break;
                }
            }
            if(flag == 0){
                WordProbability wp = entrySet.getValue();
                wp.p_w_given_R /= normFactor;
                wp.expansionWeight = wp.p_w_given_R;
                hashmap_PwGivenR_final.put(entrySet.getKey(), wp);
            }
        }
        // -- Normalizing done
        
        /* sorting list in descending order
           select top terms from the sorted list and normalize */
        list_PwGivenR = new ArrayList<>(hashmap_PwGivenR_final.values());
        hashmap_PwGivenR_final = new LinkedHashMap<>();
                
        Collections.sort(list_PwGivenR, new Comparator<WordProbability>(){
            @Override
            public int compare(WordProbability t, WordProbability t1) {
                return t.p_w_given_R<t1.p_w_given_R?1:t.p_w_given_R==t1.p_w_given_R?0:-1;
            }}); 
        // -- sorted list in descending order
        
        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR_final.get(singleTerm.w)) {
                hashmap_PwGivenR_final.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
                expansionTermCount++;
                normFactor += singleTerm.p_w_given_R;
                if(expansionTermCount>=numFeedbackTermsCausal)
                    break;
            }
        }
        //System.out.println("Final selected terms : " + hashmap_PwGivenR_final.size());
        
        // ++ Normalizing 
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR_final.entrySet()) {
            WordProbability wp = entrySet.getValue();
            wp.p_w_given_R /= normFactor;
        }
        // -- Normalizing done

        return hashmap_PwGivenR_final;
    } // end RM3_overloaded()
    
    
    private static HashMap sortTermWeight(HashMap map) {
        List<Map.Entry<String, WordProbability>> list = new ArrayList(map.entrySet());
        // Defined Custom Comparator here
        Collections.sort(list, new Comparator<Map.Entry<String, WordProbability>>() {
            @Override
            public int compare(Map.Entry<String, WordProbability> t1, Map.Entry<String, WordProbability> t2) {
                return t1.getValue().p_w_given_R<t2.getValue().p_w_given_R?1:t1.getValue().p_w_given_R==t2.getValue().p_w_given_R?0:-1;
            }
        });

        // Copying the sorted list in HashMap
        // using LinkedHashMap to preserve the insertion order
        HashMap sortedHashMap = new LinkedHashMap();
        for (Map.Entry entry : list) {
            sortedHashMap.put(entry.getKey(), entry.getValue());
        }
        return sortedHashMap;
    }
    
    
    /**
     * Returns the expanded query in BooleanQuery form with P(w|R) as 
     * corresponding weights for the expanded terms
     * @param expandedQuery The expanded query
     * @param query The query
     * @return BooleanQuery to be used for consequent re-retrieval
     * @throws Exception 
     */
    public BooleanQuery getExpandedQuery(HashMap<String, WordProbability> expandedQuery, TRECQuery query) throws Exception {

        BooleanQuery booleanQuery = new BooleanQuery();
        
        for (Map.Entry<String, WordProbability> entrySet : expandedQuery.entrySet()) {
            String key = entrySet.getKey();
            if(key.contains(":"))
                continue;
            WordProbability wProba = entrySet.getValue();
            float value = wProba.expansionWeight;
            Term t = new Term(rbcm.fieldToSearch, key);
            Query tq = new TermQuery(t);
            tq.setBoost(value);
            BooleanQuery.setMaxClauseCount(4096);
            booleanQuery.add(tq, BooleanClause.Occur.SHOULD);
        }

        return booleanQuery;
    } // ends getExpandedQuery()
    
    
    public BooleanQuery getExpandedQuery_Overloaded(HashMap<String, WordProbability> expandedQuery, String[] analyzedQuery) throws Exception {

        BooleanQuery booleanQuery = new BooleanQuery();
        
        for (Map.Entry<String, WordProbability> entrySet : expandedQuery.entrySet()) {
            String key = entrySet.getKey();
            if(key.contains(":"))
                continue;
            WordProbability wProba = entrySet.getValue();
            float value = wProba.p_w_given_R;
            wProba.expansionWeight = value;
            Term t = new Term(rbcm.fieldToSearch, key);
            Query tq = new TermQuery(t);
            tq.setBoost(value);
            BooleanQuery.setMaxClauseCount(4096);
            booleanQuery.add(tq, BooleanClause.Occur.SHOULD);
        }

        return booleanQuery;
    } // ends getExpandedQuery()
}