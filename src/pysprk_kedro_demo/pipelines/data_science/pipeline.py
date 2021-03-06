"""Example data science pipeline using PySpark.
"""

from kedro.pipeline import node, pipeline

from .nodes import train_model


def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                train_model,
                inputs=["training_data", "parameters"],
                outputs=None,
            ),
        ]
    )
