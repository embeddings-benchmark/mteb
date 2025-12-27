FAQ = """### What do aggregate measures (Rank(Borda), Mean(Task), etc.) mean?

- **Rank(borda)** is computed based on the [borda count](https://en.wikipedia.org/wiki/Borda_count), where each task is treated as a preference voter, which gives votes on the models per their relative performance on the task. The best model obtains the highest number of votes. The model with the highest number of votes across tasks obtains the highest rank. The Borda rank tends to prefer models that perform well broadly across tasks. However, given that it is a rank it can be unclear if the two models perform similarly.
- **Mean(Task)**: This is a naïve average computed across all the tasks within the benchmark. This score is simple to understand and is continuous as opposed to the Borda rank. However, the mean can overvalue tasks with higher variance in its scores.
- **Mean(TaskType)**: This is a weighted average across different task categories, such as classification or retrieval. It is computed by first computing the average by task category and then computing the average on each category. Similar to the Mean(Task) this measure is continuous and tends to overvalue tasks with higher variance. This score also prefers models that perform well across all task categories.

### What does zero-shot mean?

A model is considered zero-shot if it is not trained on any splits of the datasets used to derive the tasks.
The percentages in the table indicate what portion of the benchmark can be considered out-of-distribution for a given model.
100% means the model has not been trained on any of the datasets in a given benchmark, and therefore the benchmark score can be interpreted as the model's overall generalization performance,
while 50% means the model has been finetuned on half of the tasks in the benchmark, thereby indicating that the benchmark results should be interpreted with a pinch of salt.
This definition creates a few edge cases. For instance, multiple models are typically trained on Wikipedia title and body pairs, but we do not define this as leakage on, e.g., “WikipediaRetrievalMultilingual” and “WikiClusteringP2P” as these datasets are not based on title-body pairs.
Distilled, further fine-tunes, or in other ways, derivative models inherit the datasets of their parent models.
Based on community feedback and research findings, this definition may change in the future. Please open a PR if you notice any mistakes or want to help us refine annotations, see [GitHub](https://github.com/embeddings-benchmark/mteb/blob/06489abca007261c7e6b11f36d4844c5ed5efdcb/mteb/models/bge_models.py#L91).

### What do the other columns mean?

- **Number of Parameters**: This is the total number of parameters in the model including embedding parameters. A higher value means the model requires more CPU/GPU memory to run; thus, less is generally desirable.
- **Embedding Dimension**: This is the vector dimension of the embeddings that the model produces. When saving embeddings to disk, a higher dimension will require more space, thus less is usually desirable.
- **Max tokens**: This refers to how many tokens (=word pieces) the model can process. Generally, a larger value is desirable.
- **Zero-shot**: This indicates if the model is zero-shot on the benchmark. For more information on zero-shot see the info box above.

### Why is a model missing or not showing up?

Possible reasons why a model may not show up in the leaderboard:

- **Filter Setting**: It is being filtered out with your current filter. By default, we do not show models that are not zero-shot on the benchmark.
You can change this setting in the model selection panel.
- **Missing Results**: The model may not have been run on the tasks in the benchmark. We only display models that have been run on at least one task
in the benchmark. For visualizations that require the mean across all tasks, we only display models that have been run on all tasks in the benchmark.
You can see existing results in the [results repository](https://github.com/embeddings-benchmark/results). This is also where new results are added via PR.
- **Missing Metadata**: Currently, we only show models for which we have metadata in [mteb](https://github.com/embeddings-benchmark/mteb).
You can follow this guide on how to add a [model](https://github.com/embeddings-benchmark/mteb/blob/main/docs/adding_a_model.md) and
see existing implementations [here](https://github.com/embeddings-benchmark/mteb/tree/main/mteb/models).
"""


ACKNOWLEDGEMENT = """
<div style="border-top: 1px solid #ddd; margin-top: 30px; padding-top: 10px; font-size: 0.85em; color: #666;">
  <p><strong>Acknowledgment:</strong> We thank <a href="https://cloud.google.com/">Google</a>, <a href="https://contextual.ai/">Contextual AI</a>, <a href="https://www.laude.org/">Laude Institute</a>, <a href="https://www.servicenow.com/">ServiceNow</a> and <a href="https://huggingface.co/">Hugging Face</a> for their generous sponsorship. If you'd like to sponsor us, please get in <a href="mailto:n.muennighoff@gmail.com">touch</a>.</p>
<div class="sponsor-image-about" style="display: flex; align-items: center; gap: 10px;">
    <a href="https://cloud.google.com/">
        <img src="https://img.icons8.com/?size=512&id=17949&format=png" width="60" height="55" style="padding: 10px;">
    </a>
    <a href="https://contextual.ai/">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQd4EDMoZLFRrIjVBrSXOQYGcmvUJ3kL4U2usvjuKPla-LoRTZtLzFnb_Cu5tXzRI7DNBo&usqp=CAU" width="60" height="55" style="padding: 10px;">
    </a>
    <a href="https://www.laude.org/">
        <img src="https://media.licdn.com/dms/image/v2/D4E0BAQEf_yGxYWCZPQ/company-logo_200_200/B4EZd7LecEHcAI-/0/1750118302768/laude_institute_logo?e=2147483647&v=beta&t=HqOMQxuGeHVjZsPIZ0vJXHpR3ZBH9OVo2aYK-f9ovio" width="60" height="55" style="padding: 10px;">
    </a>
    <a href="https://www.servicenow.com/">
        <img src="https://play-lh.googleusercontent.com/HdfHZ5jnfMM1Ep7XpPaVdFIVSRx82wKlRC_qmnHx9H1E4aWNp4WKoOcH0x95NAnuYg" width="60" height="55" style="padding: 10px;">
    </a>
    <a href="https://huggingface.co">
        <img src="https://raw.githubusercontent.com/embeddings-benchmark/mteb/main/docs/images/logos/hf_logo.png" width="60" height="55" style="padding: 10px;">
    </a>
</div>

  <p style="margin-top: 5px; font-size: 0.8em;">We also thank the following companies which provide API credits to evaluate their models: <a href="https://openai.com/">OpenAI</a>, <a href="https://www.voyageai.com/">Voyage AI</a></p>
</div>
"""
