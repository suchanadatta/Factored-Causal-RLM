#!/bin/bash
# FCRLM proposed in:
# "Retrieving Potential Causes from a Query Event" --- SIGIR 2020
# Suchana Datta, Debasis Ganguly, Dwaipayan Roy, Francesca Bonin, Charles Jochim and Mandar Mitra.

cd ../

stopFilePath="/home/suchana/smart-stopwords"
if [ ! -f $stopFilePath ]
then
    echo "Please ensure that the path of the stopword-list-file is set in the .sh file."
else
    echo "Using stopFilePath="$stopFilePath
fi

if [ $# -le 7 ] 
then
    echo "Usage: " $0 " <following arguments in the order>";
    echo "1. Path of the index.";
    echo "2. Path of the query.xml file."
    echo "3. Path of the directory to store res file."
    echo "4. Number of expansion documents.";
    echo "5. Number of expansion terms: from topical set of PR.";
    echo "6. Number of expansion terms: from causal set of PR.";	
    echo "7. RM3-QueryMix (0.0-1.0): to weight between P(w|R) and P(w|Q).";
    echo "8. SimilarityFunction: 0.DefaultSimilarity, 1.BM25Similarity, 2.LMJelinekMercerSimilarity, 3.LMDirichletSimilarity.";
    exit 1;
fi

indexPath=`readlink -f $1`		# absolute address of the index
queryPath=`readlink -f $2`		# absolute address of the query file
resPath=`readlink -f $3`		# absolute directory path of the .res file
resPath=$resPath"/"

queryName=$(basename $queryPath)
prop_name="fcrlm-"$7"-"$queryName".D-"$4".topical-"$5".causal-"$6".properties"

echo "Using index at: "$indexPath
echo "Using query at: "$queryPath
echo "Using directory to store .res file: "$resPath

fieldToSearch="content"
fieldForFeedback="content"

echo "Field for searching: "$fieldToSearch
echo "Field for feedback: "$fieldForFeedback

similarityFunction=$8

case $similarityFunction in
    2) param1=0.7
       param2=0.0 ;;
    3) param1=500
       param2=0.0 ;;
esac

echo "similarity-function: "$similarityFunction" " $param1

# making the .properties file
cat > $prop_name << EOL

indexPath=$indexPath

fieldToSearch=$fieldToSearch

fieldForFeedback=$fieldForFeedback

queryPath=$queryPath

stopFilePath=$stopFilePath

resPath=$resPath

numHits= 1000
#numHits= 500

similarityFunction=$similarityFunction

param1=$param1
param2=$param2

# Number of documents
numFeedbackDocs=$4

# Number of terms in topical step
numFeedbackTermsTopical=$5

# Number of terms in causal step
numFeedbackTermsCausal=$6

rm3.queryMix=$7


EOL
# .properties file made

java -Xmx1g -cp $CLASSPATH:dist/FactoredCausalRelevanceFeedback.jar FCRLM.RelevanceBasedCausalModel $prop_name

