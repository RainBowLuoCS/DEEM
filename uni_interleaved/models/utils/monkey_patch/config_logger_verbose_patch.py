from transformers.generation.utils import GenerationConfig

   
@classmethod
def new_from_dict(cls, config_dict, **kwargs):
    """
    Instantiates a [`GenerationConfig`] from a Python dictionary of parameters.

    Args:
        config_dict (`Dict[str, Any]`):
            Dictionary that will be used to instantiate the configuration object.
        kwargs (`Dict[str, Any]`):
            Additional parameters from which to initialize the configuration object.

    Returns:
        [`GenerationConfig`]: The configuration object instantiated from those parameters.
    """
    return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
    # Those arguments may be passed along for our internal telemetry.
    # We remove them so they don't appear in `return_unused_kwargs`.
    kwargs.pop("_from_auto", None)
    kwargs.pop("_from_pipeline", None)
    # The commit hash might have been updated in the `config_dict`, we don't want the kwargs to erase that update.
    if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
        kwargs["_commit_hash"] = config_dict["_commit_hash"]

    # The line below allows model-specific config to be loaded as well through kwargs, with safety checks.
    # See https://github.com/huggingface/transformers/pull/21269
    config = cls(**{**config_dict, **kwargs})
    unused_kwargs = config.update(**kwargs)

    if return_unused_kwargs:
        return config, unused_kwargs
    else:
        return config
    
def replace_logger_verbose():
    GenerationConfig.from_dict=new_from_dict
    print('replace GenerationConfig to stop logger')