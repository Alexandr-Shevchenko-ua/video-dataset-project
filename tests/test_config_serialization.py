# STATUS: not executed, pseudocode – requires adaptation
"""Regression tests for config serialization helpers."""

import json

from config import PipelineConfig, config_to_dict


def test_config_to_dict_allows_json_dump():
    cfg = PipelineConfig()
    data = config_to_dict(cfg)
    assert isinstance(data, dict)
    # json.dumps should succeed because Paths have been stringified
    json_text = json.dumps(data)
    assert "/data" in json_text
