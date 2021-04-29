from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, split, concat
from pyspark.ml.feature import Word2Vec, Word2VecModel
import logging
import os
logging.basicConfig(level=logging.INFO)


class WordToVec:
    def __init__(self, spark):
        """
        Note:This constructor is used to initialized the instance
        Author:Aliabbas Bhojani
        """
        logging.info("Constructor of the class WordToVec is initialized")
        self.model_dir = "../models/"
        self.model_name = "word_to_vec"
        self.spark = spark

    def train_model(self, train_df, save=True):
        """
        Note:This method is used for training the model on specific dataframe
        Return: If save is true then model will save which is by default and if it is passed as false model will be returned
        Author:Aliabbas Bhojani
        """
        logging.info("Training for the Work2Vec is started")
        word2_vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="features")
        model = word2_vec.fit(train_df)
        if save:
            model.write().overwrite().save(self.model_dir+self.model_name)
            logging.info("Training for the Work2Vec is finished and model saved")
            return None
        else:
            return model

    def get_vectors_for_df(self, df):
        """
        Note:This method is used to get the vectors from the text in datafram
        Author:Aliabbas Bhojani
        """
        logging.info("Getting the vectors for the given dataframe")
        if os.path.exists(self.model_dir+self.model_name):
            model = Word2VecModel.load(self.model_dir+self.model_name+"/")
            output_df = model.transform(df)
            return output_df
        else:
            logging.info("No Models Found, retrain the model")


if __name__ == '__main__':
    spark = SparkSession.builder.appName('word_to_vec').master("spark://Sameers-MacBook-Pro.local:7077").getOrCreate()
    logging.info("Spark Session is initialised with the app name as {} and at master {}".format("word_to_vec", "spark://Sameers-MacBook-Pro.local:7077"))
    word_to_vec = WordToVec(spark)
    a_name_df = spark.read.csv("../data/A_1k_names.csv", header=True, inferSchema=True)\
        .withColumn("name", concat(col("firstname"), lit(" "), col("lastname")))\
        .select("id", "name")#split(col("name"), " ").alias("text")
    b_name_df = spark.read.csv("../data/Β_1k_names.csv", header=True, inferSchema=True)\
        .withColumn("name", concat(col("firstname"), lit(" "), col("lastname")))\
        .select("id", "name")#split(col("name"), " ").alias("text")
    df = a_name_df.unionAll(b_name_df).select(split(col("name"), " ").alias("text"))
    df.show(10, False)
    #df = a_name_df.alias("a").join(b_name_df.alias("b"), col("a.id") == col("b.id"), "inner").withColumn("concatname", concat(col("a.name"), lit(" "), col("b.name"))).select(split(col("concatname"), " ").alias("text"))
    word_to_vec.train_model(df)
    #df = word_to_vec.get_vectors_for_df(df)
    #df.show(10, False)1234
