from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from typing import List, Any, Dict
from pyspark.sql.functions import col, when, explode
from pyspark.ml.feature import StringIndexer, VectorAssembler


def transform_f(df: DataFrame) -> DataFrame:

    raw_feature_columns = ["CE_Language", "C_Language", "C_Field"]

    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in raw_feature_columns]
    pipeline = Pipeline(stages=indexers)
    df_r = pipeline.fit(df).transform(df)
    df_r = df_r.withColumn("CE_Experties_Education",when(col("CE_Experties").contains("Education Insurance"),1).otherwise(0))
    df_r = df_r.withColumn("CE_Experties_Health",when(col("CE_Experties").contains("Health Insurance"),1).otherwise(0))
    df_r = df_r.withColumn("CE_Experties_Home",when(col("CE_Experties").contains("Home Insurance"),1).otherwise(0))
    df_r = df_r.withColumn("CE_Experties_Life",when(col("CE_Experties").contains("Life Insurance"),1).otherwise(0))
    df_r = df_r.withColumn("CE_Experties_Car",when(col("CE_Experties").contains("Car Insurance"),1).otherwise(0)) 
    df_r = df_r.drop("CE_Language", "C_Language", "C_Field","CE_Experties")

    return df_r


def transform_features(data: DataFrame) -> DataFrame:

    features = ["CE_Age",
                "CE_Total_Handled_Leads",
                "CE_Successful_Leads",
                "C_Timeperiod_months",
                "C_Times_Call",
                "C_Products_Subscribed",
                "C_Premium_Products_Subscribed", 
                "C_Call_On_Same_Topic", "C_Default_Premium",
                "Call_Duration_Sec", "CE_Language_index", 
                "C_Language_index", "C_Field_index",
                "CE_Experties_Education", 
                "CE_Experties_Health",
                "CE_Experties_Home",
                "CE_Experties_Life",
                "CE_Experties_Car"
                ]

    data = data.withColumn("CE_Age",data.CE_Age.cast('float'))
    data = data.withColumn("CE_Total_Handled_Leads",data.CE_Total_Handled_Leads.cast('float'))
    data = data.withColumn("CE_Successful_Leads",data.CE_Successful_Leads.cast('float'))
    data = data.withColumn("C_Timeperiod_months",data.C_Timeperiod_months.cast('float'))
    data = data.withColumn("C_Times_Call",data.C_Times_Call.cast('float'))
    data = data.withColumn("C_Premium_Products_Subscribed",data.C_Premium_Products_Subscribed.cast('float'))
    data = data.withColumn("C_Products_Subscribed",data.C_Products_Subscribed.cast('float'))
    data = data.withColumn("C_Call_On_Same_Topic",data.C_Call_On_Same_Topic.cast('float'))
    data = data.withColumn("C_Default_Premium",data.C_Default_Premium.cast('float'))
    data = data.withColumn("Call_Duration_Sec",data.Call_Duration_Sec.cast('float'))
    data = data.withColumn("Converted",data.Converted.cast('float'))
    vector_assembler = VectorAssembler(
            inputCols=features, outputCol="features"
        )
    transformed_data = vector_assembler.transform(data).drop(*features)
    transformed_data = transformed_data.withColumn("features",col("features"))

    return transformed_data


def split_data(transformed_data: DataFrame) -> List[DataFrame]:
    
    example_test_data_ratio = 0.25
    example_train_data_ratio = 1 - example_test_data_ratio

    training_data, testing_data = transformed_data.randomSplit(
        [example_train_data_ratio, example_test_data_ratio]
    )

    

    return [training_data, testing_data]