import os

def get_current_backend():
    return os.environ.get("CUDF_POLARS__BACKEND", "cudf")


excluded_ops = [
    # "select",
    "reduce",
    "group_by",
    "rolling",
    "cond_join",
    "join",
    "empty",
    "hconcat",
    "mapfn",
    # "union",
    # "filter",
    "projection",
    # "slice",
    "sort",
    # "hstack",
    "distinct"
]

def is_triton_backend(op: str) -> bool:
    back = get_current_backend()
    if back == "combined_triton":
        return op not in excluded_ops
            
    return back == "triton"