"""Example nodes to solve some common data science problems using PySpark,
such as:
* Train a machine learning model on a training dataset
* Make predictions using the model
* Evaluate the model based on its prediction
"""
import logging
import mlflow
from mlflow import sklearn
import datetime as dt
from typing import Any, Dict
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame


def train_model(training_data: DataFrame, parameters: Dict[str, Any]) -> RandomForestClassifier:
    """Node for training a random forest model to classify the data.
    The number of trees is defined in conf/project/parameters.yml
    and passed into this node via the `parameters` argument.
    For more information about random forest classifier with spark, please visit:
    https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier
    """
    experiment_id = mlflow.create_experiment("Agent_Routing_Pyspark" + str(dt.datetime.now()))
    experiment = mlflow.get_experiment(experiment_id) 
    with mlflow.start_run():
        
        classifier = RandomForestClassifier(featuresCol = 'features', labelCol = 'Converted')
        rfModel = classifier.fit(training_data)
        sklearn.log_model(
                    sk_model=rfModel, artifact_path="model"
                )
        
        print('*******************Training Finished*******************')

