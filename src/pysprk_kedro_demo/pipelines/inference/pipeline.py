"""Example data science pipeline using PySpark.
"""

from kedro.pipeline import node, pipeline

from .nodes import get_inference


def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                get_inference,
                inputs="infer_data",
                outputs="predictions",
            ),
        ]
    )
