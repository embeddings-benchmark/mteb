---
{{ card_data }}
---

# {{ benchmark_name }}

<div align="center" style="padding: 40px 20px; background-color: white; border-radius: 12px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); max-width: 600px; margin: 0 auto;">
  <h1 style="font-size: 3.5rem; color: #1a1a1a; margin: 0 0 20px 0; letter-spacing: 2px; font-weight: 700;">{{ benchmark_name }}</h1>
  <div style="font-size: 1.5rem; color: #4a4a4a; margin-bottom: 5px; font-weight: 300;">An <a href="https://github.com/embeddings-benchmark/mteb" style="color: #2c5282; font-weight: 600; text-decoration: none;" onmouseover="this.style.textDecoration='underline'" onmouseout="this.style.textDecoration='none'">MTEB</a> benchmark</div>
  <div style="font-size: 0.9rem; color: #2c5282; margin-top: 10px;">Massive Text Embedding Benchmark</div>
</div>

{% if benchmark_description %}
{{ benchmark_description }}
{% endif %}

## Tasks

| Task | Type | Description |
|------|------|-------------|
{% for task in tasks -%}
| {% if task.reference %}[{{ task.name }}]({{ task.reference }}){% else %}{{ task.name }}{% endif %} | {{ task.simplified_type }} | {{ task.description }} |
{% endfor %}

## How to evaluate on this benchmark

```python
import mteb

benchmark = mteb.get_benchmark("{{ benchmark_name }}")
model = mteb.get_model(YOUR_MODEL)
results = mteb.run(model, tasks=benchmark)
```

{% if citation %}
## Citation

```bibtex
{{ citation }}
```
{% endif %}

---
*This benchmark card was automatically generated using [MTEB](https://github.com/embeddings-benchmark/mteb)*
