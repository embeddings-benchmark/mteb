from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from mteb.abstasks import AbsTaskMultilabelClassification
from mteb.abstasks.task_metadata import TaskMetadata


# NOTE: In the paper, this is grouped with linear probe tasks.
# See https://github.com/embeddings-benchmark/mteb/pull/2035#issuecomment-2661626309.
class VOC2007Classification(AbsTaskMultilabelClassification):
    metadata = TaskMetadata(
        name="VOC2007",
        description="Classifying bird images from 500 species.",
        reference="http://host.robots.ox.ac.uk/pascal/VOC/",
        dataset={
            "path": "mteb/VOC2007",
            "revision": "96d47458fb2f40713997b0ea271688c02bac840b",
        },
        type="ImageClassification",
        category="i2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="lrap",
        date=(
            "2005-01-01",
            "2014-01-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@article{Everingham10,
  author = {Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.},
  journal = {International Journal of Computer Vision},
  month = jun,
  number = {2},
  pages = {303--338},
  title = {The Pascal Visual Object Classes (VOC) Challenge},
  volume = {88},
  year = {2010},
}
""",
    )

    # Override default column name in the subclass
    label_column_name: str = "classes"

    # To be removed when we want full results
    n_experiments: int = 5
    input_column_name: str = "image"
    evaluator = MultiOutputClassifier(estimator=LogisticRegression())
