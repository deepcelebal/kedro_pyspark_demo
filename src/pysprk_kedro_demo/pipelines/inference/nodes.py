import mlflow


def get_inference(testing_data: DataFrame) -> DataFrame:
    
    logged_model = 'runs:/873d89ca676246c691c1a3c0ce901104/model_rf'

    # Load model as a Spark UDF. Override result_type if the model does not return double values.
    loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='double')
    predictions = loaded_model.transform(testing_data)

    return predictions