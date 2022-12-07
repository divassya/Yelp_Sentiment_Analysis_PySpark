import sys
import re
import numpy as np
import pyspark 
from pyspark.sql import SQLContext
import os
from numpy import dot
from operator import add
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.functions import lower, col
from pyspark.sql.functions import *
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
import spacy
import contractions
import os
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC

spark = SparkSession.builder\
          .appName("SentimentAnalysis")\
          .getOrCreate()

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)
dfTextTarget = sqlContext.read.json(sys.argv[1])

# dfTextTarget = dfTextTarget.sample(0.001)
project_folder = sys.argv[2]

def sentiment(stars):
  if stars <= 3:
    return 0
  else:
    return 1
nlp = spacy.load('en_core_web_sm')

def text_preprocess(s):
  s = " ".join([contractions.fix(w) for w in s.split()])
  doc = nlp(s)
  tokens = [token.text.lower() for token in doc if token.text.isalpha()]
  return " ".join(tokens)

func_udf = udf(sentiment, IntegerType())
dfTextTarget = dfTextTarget.withColumn("target", func_udf(dfTextTarget['stars']))
dfTextTarget = dfTextTarget.select('text','target')

text_preprocess_udf = udf(text_preprocess, StringType())
dfTextTarget = dfTextTarget.withColumn('text', text_preprocess_udf(col('text')))
dfTextTarget = dfTextTarget.select('text', 'target')

"""## Data Split"""

(train_set, test_set) = dfTextTarget.randomSplit([0.99, 0.01], seed = 2000)
# (train_set, test_set) = dfTextTarget.randomSplit([0.8, 0.2], seed = 2000)


def eval_model(model_name,model):

  tokenizer = Tokenizer(inputCol="text", outputCol="words")
  hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
  idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
  label_stringIdx = StringIndexer(inputCol = "target", outputCol = "label")
  pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx,model])
  pipelineFit = pipeline.fit(train_set)

  predictions_train = pipelineFit.transform(train_set)
  predictions_test = pipelineFit.transform(test_set)

  train_accuracy = predictions_train.filter(predictions_train.label == predictions_train.prediction).count() / float(train_set.count())
  test_accuracy = predictions_test.filter(predictions_test.label == predictions_test.prediction).count() / float(test_set.count())

  evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
  train_roc_auc = evaluator.evaluate(predictions_train)
  test_roc_auc = evaluator.evaluate(predictions_test)
  metricsList = [(model_name,train_accuracy,test_accuracy,train_roc_auc,test_roc_auc)]
  return metricsList

"""## TFIDF + LInear SVC"""
lsvc = LinearSVC(maxIter=10, regParam=0.1)
lsvc_metricsList = eval_model('LinearSVC', lsvc)
spark.createDataFrame(lsvc_metricsList).write.csv(project_folder+'metrics')
