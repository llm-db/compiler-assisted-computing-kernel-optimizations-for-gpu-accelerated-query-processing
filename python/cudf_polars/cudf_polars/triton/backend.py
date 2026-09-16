import os

def get_current_backend():
    return os.environ.get("CUDF_POLARS__BACKEND", "cudf")


def is_triton_backend() -> bool:
    return get_current_backend() == "triton"