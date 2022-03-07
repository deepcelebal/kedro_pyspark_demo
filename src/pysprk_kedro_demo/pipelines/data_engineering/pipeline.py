"""Example data engineering pipeline with PySpark.
"""

from kedro.pipeline import node, pipeline

from .nodes import transform_f, transform_features, split_data


def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                transform_f,
                inputs="agent_routing",
                outputs="agent_routing_v1",
            ),
            node(
                transform_features,
                inputs="agent_routing_v1",
                outputs="agent_routing_v2",
            ),
            node(
                split_data,
                inputs="agent_routing_v2",
                outputs=["training_data", "testing_data"],
            ),
        ]
    )
