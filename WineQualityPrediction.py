import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pyspark
from pyspark import sql
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark=SparkSession.builder.appName('wine_prediction').getOrCreate()

spark

train_data= spark.read.format("com.databricks.spark.csv").csv('s3://aws-logs-471112668218-us-east-1/winequality/TrainingDataset.csv', header=True, sep=";")
train_data.printSchema()

test_data= spark.read.format("com.databricks.spark.csv").csv('s3://aws-logs-471112668218-us-east-1/winequality/ValidationDataset.csv', header=True, sep=";")
test_data.printSchema()

old_column_name = train_data.schema.names
print(train_data.schema)
clean_column_name = []

for name in old_column_name:
    clean_column_name.append(name.replace('"',''))

old_column_name = test_data.schema.names
print(test_data.schema)
clean_column_name = []

for name in old_column_name:
    clean_column_name.append(name.replace('"',''))

from pyspark.rdd import reduce
train_data = reduce(lambda train_data, idx: train_data.withColumnRenamed(old_column_name[idx], clean_column_name[idx]), range(len(clean_column_name)), train_data)

test_data = reduce(lambda test_data, idx: test_data.withColumnRenamed(old_column_name[idx], clean_column_name[idx]), range(len(clean_column_name)), test_data)

train_data.groupby("quality").count().show()

test_data.groupby("quality").count().show()

train_data.toPandas()

test_data.toPandas()

from pyspark.sql.functions import col

train_data.show()

from pyspark.sql.functions import isnull, when, count
train_data.select([count(when(isnull(c), c)).alias(c) for c in train_data.columns]).show()

train_dataset = train_data.select(col('fixed acidity').cast('float'),col('volatile acidity').cast('float'),col('citric acid').cast("float"),col('residual sugar').cast('float'),col('chlorides').cast('float'),col('free sulfur dioxide').cast('float'),col("total sulfur dioxide").cast('float'),col('density').cast('float'),col('pH').cast('float'),col('sulphates').cast('float'),col('alcohol').cast('float'),col("quality").cast("float"))
train_dataset.show()

test_dataset = test_data.select(col('fixed acidity').cast('float'),col('volatile acidity').cast('float'),col('citric acid').cast("float"),col('residual sugar').cast('float'),col('chlorides').cast('float'),col('free sulfur dioxide').cast('float'),col("total sulfur dioxide").cast('float'),col('density').cast('float'),col('pH').cast('float'),col('sulphates').cast('float'),col('alcohol').cast('float'),col("quality").cast("float"))
test_dataset.show()

required_features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=required_features, outputCol='features')
training_data = assembler.transform(train_dataset)

testing_data = assembler.transform(test_dataset)

train_rows = training_data.count()
test_rows = testing_data.count()
print("Training Rows:", train_rows)
print("Testing Rows:", test_rows)

training_data.select("features").show(truncate=False)

lr = LogisticRegression(labelCol="quality",featuresCol="features",maxIter=10,regParam=0.3)

model = lr.fit(training_data)

predictions_train = model.transform(training_data)

predictions_test = model.transform(testing_data)

model.write().overwrite().save("s3://aws-logs-471112668218-us-east-1/winequality/LogisticRegression.model")

predictions_test.select("prediction").show()

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol='quality',predictionCol='prediction')

F1score = evaluator.evaluate(predictions_train, {evaluator.metricName: "f1"})
accuracy = evaluator.evaluate(predictions_train, {evaluator.metricName: "accuracy"})
print('Train Accuracy = ', accuracy)
print('F1 Score = ', F1score)

F1score = evaluator.evaluate(predictions_test, {evaluator.metricName: "f1"})
accuracy = evaluator.evaluate(predictions_test, {evaluator.metricName: "accuracy"})
print('Testing Accuracy = ', accuracy)
print('F1 Score = ', F1score)

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol='quality',featuresCol='features',maxDepth=5,numTrees=100)

model = rf.fit(training_data)

predictions_train = model.transform(training_data)

predictions_test = model.transform(testing_data)

model.write().overwrite().save("s3://aws-logs-471112668218-us-east-1/winequality/RandomForestClassifier.model")

predictions_test.select("prediction").show()

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol='quality',
    predictionCol='prediction')

F1score = evaluator.evaluate(predictions_train, {evaluator.metricName: "f1"})
accuracy = evaluator.evaluate(predictions_train, {evaluator.metricName: "accuracy"})
print('Train Accuracy = ', accuracy)
print('F1 Score = ', F1score)

F1score = evaluator.evaluate(predictions_test, {evaluator.metricName: "f1"})
accuracy = evaluator.evaluate(predictions_test, {evaluator.metricName: "accuracy"})
print('Testing Accuracy = ', accuracy)
print('F1 Score = ', F1score)





testing_data.show()

pred=model.transform(testing_data)

pred.count()

pred1 = pred.select(col('quality').cast('float'),col('prediction').cast('float'))

# Show the results
pred1.show()

