# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Syntec: 100 questions for information retrieval"""
import json
import datasets

_DESCRIPTION = """\
This dataset is based on the Syntec collective bargaining agreement. Its purpose is information retrieval.
"""

_SPLITS = ["documents", "queries"]
_HOMEPAGE = "https://huggingface.co/datasets/lyon-nlp/mteb-fr-retrieval-syntec-s2p"
_LICENSE = "Creative Commons Attribution Non Commercial Share Alike 4.0 International"
_URLS = {
    split: f"https://huggingface.co/datasets/lyon-nlp/mteb-fr-retrieval-syntec-s2p/resolve/main/{split}.json"\
    for split in _SPLITS
}

class Syntec(datasets.GeneratorBasedBuilder):
    """Syntec: 100 questions for information retrieval"""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="documents", version=VERSION, description="Corpus of articles from the Syntec collective bargaining agreement"),
        datasets.BuilderConfig(name="queries", version=VERSION, description="Corpus of 100 manually annotated queries"),
    ]

    # Avoid setting default config so that an error is raised asking the user
    # to specify the piece of the dataset wanted
    DEFAULT_CONFIG_NAME = "documents"

    def _info(self):
        if self.config.name == "documents":
            features = {
                "section": datasets.Value("string"),
                "id": datasets.Value("string"),
                "title": datasets.Value("string"),
                "content": datasets.Value("string"),
                "url": datasets.Value("string"),
            }
        elif self.config.name == "queries":
            features = {
                "Question": datasets.Value("string"),
                "Article": datasets.Value("string"),
            }
        else:
            raise ValueError(f"Please specify a valid config name : {_SPLITS}")
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        if self.config.name == "documents":
            dl_path = dl_manager.download_and_extract(_URLS["documents"])
            return [datasets.SplitGenerator(name="documents", gen_kwargs={"filepath": dl_path})]
        elif self.config.name == "queries":
            dl_paths = dl_manager.download_and_extract(_URLS["queries"])
            return [datasets.SplitGenerator(name="queries", gen_kwargs={"filepath": dl_paths})]
        else:
            raise ValueError(f"Please specify a valid config name : {_SPLITS}")


    def _generate_examples(self, filepath):
        if self.config.name in ["documents", "queries"]:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
                for key, row in enumerate(data):
                    if self.config.name == "documents":
                        features = {
                            "section": row["section"],
                            "id": row["id"],
                            "title": row["title"],
                            "content": row["content"],
                            "url": row["url"]
                        }
                    elif self.config.name == "queries":
                        features = {
                            "Question": row["Question"],
                            "Article": row["Article"],
                        }
                    else:
                        raise ValueError(f"Please specify a valid config name : {_SPLITS}")
                    yield key, features
        else:
            raise ValueError(f"Please specify a valid config name : {_SPLITS}")
