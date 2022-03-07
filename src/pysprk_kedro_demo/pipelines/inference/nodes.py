
import logging
import mlflow
from mlflow import spark
import datetime as dt
from typing import Any, Dict
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame



def get_inference(testing_data: DataFrame) -> DataFrame:
    
    logged_model = 'runs:/873d89ca676246c691c1a3c0ce901104/model_rf'

    # Load model as a Spark UDF. Override result_type if the model does not return double values.
    loaded_model = mlflow.spark.load_model(logged_model)
    predictions = loaded_model.transform(testing_data)

    return predictions