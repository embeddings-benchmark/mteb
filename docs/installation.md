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
pip install mteb[openai]
```

If a specific model requires a dependency it will raise an error with the recommended installation. 

<!-- TODO: add this 
To get an overview of the implemented models see [here](missing). 
-->
