
import logging
import mlflow
from mlflow import spark
import datetime as dt
from typing import Any, Dict
from pyspark.sql import DataFrame
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pysprk_kedro_demo.pipelines import data_engineering
from pysprk_kedro_demo.pipelines.data_engineering.nodes import transform_f, transform_features


def get_inference(testing_data: DataFrame):
    
    testing_data = transform_f(testing_data)
    testing_data = transform_features(testing_data)

    logged_model = 'runs:/873d89ca676246c691c1a3c0ce901104/model_rf'

    loaded_model = mlflow.spark.load_model(logged_model)
    predictions = loaded_model.transform(testing_data)
    predict = predictions.select(predictions.prediction)

    return predictions