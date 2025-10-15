# Installation

Installation is as simple as:

=== "pip"
    ```bash
    pip install mteb
    ```

=== "uv"
    ```bash
    uv add mteb
    ```

## Model Specific Installations

If you want to run certain models implemented within mteb you will often need some additional dependencies. These can be installed using:

```bash
pip install mteb[cohere]
```

If a specific model requires a dependency it will raise an error with the recommended installation. To see full list of available models you can look at the [models overview](./overview/available_models/text.md).
