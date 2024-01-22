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
"""Alloprof: a new French question-answer education dataset and its use in an information retrieval case study"""
import json
import datasets


_CITATION = """\
@misc{lef23,
  doi = {10.48550/ARXIV.2302.07738},
  url = {https://arxiv.org/abs/2302.07738},
  author = {Lefebvre-Brossard, Antoine and Gazaille, Stephane and Desmarais, Michel C.},
  keywords = {Computation and Language (cs.CL), Information Retrieval (cs.IR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Alloprof: a new French question-answer education dataset and its use in an information retrieval case study},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
"""

_DESCRIPTION = """\
This is a re-edit from the Alloprof dataset (which can be found here : https://huggingface.co/datasets/antoinelb7/alloprof).

For more information about the data source and the features, please refer to the original dataset card made by the authors, along with their paper available here : https://arxiv.org/abs/2302.07738

This re-edition of the dataset has been made for easier usage in the MTEB benchmarking pipeline. (https://huggingface.co/spaces/mteb/leaderboard). It is a filtered version of the original dataset, in a more ready-to-use format.
"""

_SPLITS = ["documents", "queries-train", "queries-test"]
_HOMEPAGE = "https://huggingface.co/datasets/antoinelb7/alloprof"
_LICENSE = "Creative Commons Attribution Non Commercial Share Alike 4.0 International"
_URLS = {
    split: f"https://huggingface.co/datasets/lyon-nlp/alloprof/resolve/main/{split}.json"\
    for split in _SPLITS
}

class Alloprof(datasets.GeneratorBasedBuilder):
    """Alloprof: a new French question-answer education dataset and its use in an information retrieval case study"""

    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="documents", version=VERSION, description="Corpus of documents from the Alloprof website"),
        datasets.BuilderConfig(name="queries", version=VERSION, description="Corpus of queries from students"),
    ]

    # Avoid setting default config so that an error is raised asking the user
    # to specify the piece of the dataset wanted
    DEFAULT_CONFIG_NAME = "documents"

    def _info(self):
        if self.config.name == "documents":
            features = {
                "uuid": datasets.Value("string"),
                "title": datasets.Value("string"),
                "topic": datasets.Value("string"),
                "text": datasets.Value("string"),
            }
        elif self.config.name == "queries":
            features = {
                "id": datasets.Value("int32"),
                "text": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "relevant": datasets.Sequence(datasets.Value("string")),
                "subject": datasets.Value("string"),
            }
        else:
            raise ValueError(f"Please specify a valid config name : {_SPLITS}")
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.name == "documents":
            dl_path = dl_manager.download_and_extract(_URLS["documents"])
            return [datasets.SplitGenerator(name="documents", gen_kwargs={"filepath": dl_path})]
        elif self.config.name == "queries":
            dl_path_train = dl_manager.download_and_extract(_URLS["queries-train"])
            dl_path_test = dl_manager.download_and_extract(_URLS["queries-test"])
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": dl_path_train}),
                    datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": dl_path_test})]
        else:
            raise ValueError(f"Please specify a valid config name : {_SPLITS}")


    def _generate_examples(self, filepath):
        if self.config.name in ["documents", "queries"]:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
                for key, row in enumerate(data):
                    if self.config.name == "documents":
                        features = {
                            "uuid": row["uuid"],
                            "title": row["title"],
                            "topic": row["topic"],
                            "text": row["text"],
                        }
                    elif self.config.name == "queries":
                        features = {
                            "id": row["id"],
                            "text": row["text"],
                            "answer": row["answer"],
                            "relevant": row["relevant"],
                            "subject": row["subject"],
                        }
                    else:
                        raise ValueError(f"Please specify a valid config name : {_SPLITS}")
                    yield key, features
        else:
            raise ValueError(f"Please specify a valid config name : {_SPLITS}")
