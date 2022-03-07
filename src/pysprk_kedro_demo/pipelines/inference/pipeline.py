"""Example data science pipeline using PySpark.
"""

from kedro.pipeline import node, pipeline

from .nodes import get_inference


def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                get_inference,
                inputs="testing_data",
                outputs="predictions",
            ),
        ]
    )
