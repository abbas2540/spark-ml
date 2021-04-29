from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import Word2Vec, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    spark = SparkSession.builder.appName('word_to_vec').master("spark://Sameers-MacBook-Pro.local:7077").getOrCreate()
    stage_1 = Word2Vec(vectorSize=3, minCount=0, inputCol="text_a", outputCol="features_a_index")
    stage_2 = Word2Vec(vectorSize=3, minCount=0, inputCol="text_b", outputCol="features_b_index")
    stage_3 = VectorAssembler(inputCols=["features_a_index", "features_b_index"], outputCol="features")
    pipeline = Pipeline(stages=[stage_1, stage_2, stage_3])
    a_name_df = spark.read.csv("../data/A_1k_names.csv", header=True, inferSchema=True).withColumn("name", concat(
        col("firstname"), lit(" "), col("lastname"))).select("id", split(col("name"), " ").alias("text_a"))
    b_name_df = spark.read.csv("../data/Β_1k_names.csv", header=True, inferSchema=True).withColumn("name", concat(
        col("firstname"), lit(" "), col("lastname"))).select("id", split(col("name"), " ").alias("text_b"))
    joined_df = a_name_df.alias("a").crossJoin(b_name_df.alias("b")).withColumn("label", when(col("a.id") == col("b.id"), lit("1")).otherwise(lit("0"))).selectExpr("label", "text_a", "text_b")
    matched_df = joined_df.where(col("label") == 1)
    not_matched_df = joined_df.where(col("label") == 0).limit(1000)
    labeled_df = matched_df.unionAll(not_matched_df)
    labeled_df.show(10, False)
    pipeline_model = pipeline.fit(labeled_df)
    transform_df = pipeline_model.transform(labeled_df).selectExpr("cast(label as double) label", "features")
    # view the transformed data
    (train_df, test_df) = transform_df.randomSplit([0.7, 0.3], 24)
    logging.info("Count of training data: {}".format(train_df.count()))
    logging.info("Count of testing data: {}".format(test_df.count()))
    svm = LinearSVC(maxIter=5, regParam=0.01)
    model = svm.fit(train_df)
    logging.info("Model Coefficient {}".format(model.coefficients))
    logging.info("Model Intercept {}".format(model.intercept))
    logging.info("Model number of classes {}".format(model.numClasses))
    logging.info("Model number of features {}".format(model.numFeatures))
    predictions = model.transform(test_df)
    evaluator_svm = BinaryClassificationEvaluator(rawPredictionCol="prediction")
    area_under_curve = evaluator_svm.evaluate(predictions)
    logging.info("Area Under Curve is {}".format(area_under_curve))
    new_df = spark.createDataFrame([("ALIABBAS BHOJANI", "LIABBAS BHOJANI"), ("ALIABBAS BHOJANI", "MUSTAFA CHALLAWALA")]).toDF("text_a", "text_b").select(split(col("text_a"), " ").alias("text_a"), split(col("text_b"), " ").alias("text_b"))
    new_trans_df = pipeline_model.transform(new_df)
    model.transform(new_trans_df).show(5, False)