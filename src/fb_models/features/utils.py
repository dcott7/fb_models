from collections import defaultdict


def convert_default_dict(obj):
    if isinstance(obj, dict):
        return {k: convert_default_dict(v) for k, v in obj.items()}
    return obj