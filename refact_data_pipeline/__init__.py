import importlib
from typing import Union

from refact_data_pipeline.datadef import DatasetDef, DatasetMix, DatasetOpts


def find_dataset_def(dsname: str) -> Union[DatasetDef, DatasetMix]:
    submod_name, name = dsname.split(":")
    mod = importlib.import_module(f"data_pipeline.{submod_name}")
    assert (
        name in mod.__dict__
    ), f"dataset '{name}' was not found in '{submod_name}'"
    f = getattr(mod, name)
    assert callable(f)
    d = f()
    assert isinstance(d, (DatasetDef, DatasetMix))
    return d
