"""Reference models registry for MTEB.

This module defines the reference models that should be consistently evaluated
across MTEB benchmarks to ensure baseline coverage and comparability.

Reference models are selected based on:
- Representative coverage across model types (static, encoders, decoders)
- Representative coverage across model sizes
- Representative coverage across training approaches
- Multilingual capability where possible
- Practical relevance and common usage

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mteb.abstasks.AbsTask import AbsTask

# Core reference models that should be evaluated on all relevant benchmarks
REFERENCE_MODELS = [
    "intfloat/multilingual-e5-small",
    "sentence-transformers/static-similarity-mrl-multilingual-v1", 
    "minishlab/potion-multilingual-128M",
    "bm25s",
]

# Models that should only be evaluated on specific task types
TASK_TYPE_RESTRICTIONS = {
    "bm25s": ["Retrieval"], 
}

def get_reference_models() -> list[str]:
    """Get the complete list of reference model names.
    
    Returns:
        List of all reference model names
    """
    return REFERENCE_MODELS.copy()

def get_reference_models_for_task_type(task_type: str) -> list[str]:
    """Get reference models that should be evaluated on a specific task type.
    
    Args:
        task_type: The task type (e.g., "Retrieval", "Classification", "STS", etc.)
        
    Returns:
        List of model names that should be evaluated on this task type
    """
    applicable_models = []
    
    for model in REFERENCE_MODELS:
        # Check if model is restricted to specific task types
        if model in TASK_TYPE_RESTRICTIONS:
            if task_type in TASK_TYPE_RESTRICTIONS[model]:
                applicable_models.append(model)
        else:
            # Model applies to all task types
            applicable_models.append(model)
    
    return applicable_models

def get_reference_models_for_task(task: AbsTask) -> list[str]:
    """Get reference models that should be evaluated on a specific task.
    
    Args:
        task: The MTEB task instance
        
    Returns:
        List of model names that should be evaluated on this task
    """
    # Get task type from the task metadata, which is more reliable
    # than trying to parse class names
    task_type = task.metadata.type
    return get_reference_models_for_task_type(task_type)

def is_reference_model(model_name: str) -> bool:
    """Check if a model is a reference model.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if the model is a reference model, False otherwise
    """
    return model_name in REFERENCE_MODELS

def get_task_type_restrictions() -> dict[str, list[str]]:
    """Get the task type restrictions for reference models.
    
    Returns:
        Dictionary mapping model names to their allowed task types
    """
    return TASK_TYPE_RESTRICTIONS.copy()