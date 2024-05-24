from . import sentence_transformers_models

from mteb.model_meta import ModelMeta


def get_model(model_name: str) -> ModelMeta:
    """A function to fetch a model metadata object by name.

    Args:
        model_name: Name of the model to fetch
    """
    return models[model_name]


model_modules = [sentence_transformers_models]
models = {}


for module in model_modules:
    for mdl in module.__dict__.values():
        if isinstance(mdl, ModelMeta):
            models[mdl.name] = mdl
