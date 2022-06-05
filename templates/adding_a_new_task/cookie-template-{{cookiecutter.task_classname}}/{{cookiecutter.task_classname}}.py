from ...abstasks import AbsTaskClassification, MultilingualTask

_LANGUAGES = {{ cookiecutter.eval_langs.l }}


class {{ cookiecutter.task_classname }}(MultilingualTask, AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "{{ cookiecutter.name }}",
            "hf_hub_name": "{{ cookiecutter.hf_hub_name }}",
            "description": "{{ cookiecutter.description }}",
            "reference": "{{ cookiecutter.reference }}",
            "category": "{{ cookiecutter.category }}",
            "type": "{{ cookiecutter.type }}",
            "eval_splits": {{ cookiecutter.eval_splits.l }},
            "eval_langs": _LANGUAGES,
            "main_score": "{{ cookiecutter.main_score }}",
        }
