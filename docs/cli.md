# CLI

<!-- 
We essentially just need to make this cli.py's docstring -- figure out a way to do this automatically

We can then extend it to be more detailed going forward. Ideally adding some documentation on the different arguments
-->



## Using multiple GPUs

Using multiple GPUs in parallel can be done by just having a [custom encode function](missing) that distributes the inputs to multiple GPUs like e.g. [here](https://github.com/microsoft/unilm/blob/b60c741f746877293bb85eed6806736fc8fa0ffd/e5/mteb_eval.py#L60) or [here](https://github.com/ContextualAI/gritlm/blob/09d8630f0c95ac6a456354bcb6f964d7b9b6a609/gritlm/gritlm.py#L75).
