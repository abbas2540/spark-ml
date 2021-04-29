from pyspark.ml.classification import LinearSVC, LinearSVCModel
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import concat, col, lit, split, when
from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import StringType, ArrayType
from python.src.word_to_vec import WordToVec
import logging
import os
logging.basicConfig(level=logging.INFO)


class SparkSVM:
    def __init__(self, spark, word_to_vec):
        """
        Note:
        Author: Aliabbas Bhojani
        """
        logging.info("Constructor of the class WordToVec is initialized")
        self.model_dir = "../models/"
        self.model_name = "svm"
        self.spark = spark
        self.word_to_vec = word_to_vec
        
    def train_model(self, train_df, save=True):
        """
        Note:
        Author: Aliabbas Bhojani
        """
        logging.info("Training the model with spark context as {}".format(spark))
        sc = spark.sparkContext
        df = sc.parallelize([Row(label=1.0, features=Vectors.dense(1.0, 1.0, 1.0)),Row(label=0.0, features=Vectors.dense(1.0, 2.0, 3.0))]).toDF()
        df.printSchema()
        train_df.printSchema()
        svm = LinearSVC(maxIter=5, regParam=0.01)
        model = svm.fit(train_df)
        print(model.coefficients)
        print(model.intercept)
        print(model.numClasses)
        print(model.numFeatures)
        if save:
            model.write().overwrite().save(self.model_dir+self.model_name)
            logging.info("Training for the Work2Vec is finished and model saved")
            return None
        else:
            return model

    def get_output(self, test_df):
        """
        Note:
        Author: Aliabbas Bhojani
        """
        test_df = self.word_to_vec.get_vectors_for_df(test_df).selectExpr("result as features")
        if os.path.exists(self.model_dir+self.model_name):
            model = LinearSVCModel.load(self.model_dir+self.model_name+"/")
            result = model.transform(test_df)
            result.show(10, False)
            print(result.prediction)
            print(result.rawPrediction)
        '''svm_path = temp_path + "/svm"
        svm.save(svm_path)
        svm2 = LinearSVC.load(svm_path)
        svm2.getMaxIter()
        model_path = temp_path + "/svm_model"
        model.save(model_path)
        model2 = LinearSVCModel.load(model_path)
        model.coefficients[0] == model2.coefficients[0]
        model.intercept == model2.intercept'''
        return None


udf1 = F.udf(lambda x, y: Vectors.dense(x, y), )

if __name__ == '__main__':
    spark = SparkSession.builder.appName('spark-ml').master("spark://Sameers-MacBook-Pro.local:7077").getOrCreate()
    word_to_vec = WordToVec(spark)
    spark_svm = SparkSVM(spark, word_to_vec)
    a_name_df = spark.read.csv("../data/A_1k_names.csv", header=True, inferSchema=True).withColumn("name", concat(
        col("firstname"), lit(" "), col("lastname"))).select("id", split(col("name"), " ").alias("text"))
    a_vector_df = word_to_vec.get_vectors_for_df(a_name_df)
    b_name_df = spark.read.csv("../data/Β_1k_names.csv", header=True, inferSchema=True).withColumn("name", concat(
        col("firstname"), lit(" "), col("lastname"))).select("id", split(col("name"), " ").alias("text"))
    b_vector_df = word_to_vec.get_vectors_for_df(b_name_df)
    #joined_df = a_name_df.alias("a").crossJoin(b_name_df.alias("b")).withColumn("label", when(col("a.id") == col("b.id"), lit("1")).otherwise(lit("0"))).withColumn("concatname", concat(col("a.name"), lit(" "), col("b.name"))).select("label", split(col("concatname"), " ").alias("text"))
    #matched_df = joined_df.where(col("label") == 1)
    #not_matched_df = joined_df.where(col("label") == 0).limit(1000)
    #labeled_df = matched_df.unionAll(not_matched_df)
    #transformed_df = word_to_vec.get_vectors_for_df(labeled_df).selectExpr("cast(label as double) label", "result as features")
    #transformed_df.show(10, False)
    #spark_svm.train_model(transformed_df)
    #test_df = spark.createDataFrame(data=[("ALIABBAS BHOJANI", "@$%#$^#& 12435")]).toDF("nameA", "nameB").withColumn("concatname", concat(col("nameA"), lit(" "), col("nameB"))).select(split(col("concatname"), " ").alias("text"))
    #spark_svm.get_output(test_df)
    df = a_vector_df.alias("a").crossJoin(b_vector_df.alias("b")).withColumn("label", when(col("a.id") == col("b.id"), lit("1")).otherwise(lit("0")))
    df.selectExpr("a.id", "a.features as featuresA", "b.features as featuresB").rdd.map(lambda x: (x["id"], Vectors.dense(x["featuresA"], x["featuresB"]))).toDF(["id", "features"]).show(10, False)
        #.withColumn("joinedf", Vectors.dense(col("a.features"), col("b.features"))).show(10, False)
    #    .selectExpr("label", "concatresult as features").show(10, False)
    '''sc = spark.sparkContext
    df = sc.parallelize([Row(label=1.0, features=Vectors.dense(Vectors.dense(1.0, 1.0, 1.0), Vectors.dense(2.0, 2.0, 2.0))), Row(label=0.0, features=Vectors.dense(Vectors.dense(1.0, 2.0, 3.0), Vectors.dense(3.0, 2.0, 1.0)))]).toDF()
    df.show()
    svm = LinearSVC(maxIter=5, regParam=0.01)
    model = svm.fit(df)
    print(model.coefficients)
    print(model.intercept)
    print(model.numClasses)
    print(model.numFeatures)
    test0 = sc.parallelize([Row(features=Vectors.dense(Vectors.dense(5.0, 5.0, 5.0), Vectors.dense(6.0, 6.0, 6.0)))]).toDF()
    result = model.transform(test0).head()
    print(result.prediction)
    print(result.rawPrediction)'''
