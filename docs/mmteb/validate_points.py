from __future__ import annotations

import logging
import os

from jsonlines import Reader
from pydantic import BaseModel, ConfigDict, Field, ValidationError, conint, constr

commit_exceptions = {
    "scores_from_old_system",
    # <100 points: from before max points were enforced
    "440",
    "543",
    "616",
    "636",
    # >100 points: from before max points were enforced (reduced to 100 points)
    "583",
}


# Define a Pydantic model to represent each JSON object
class JsonObject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    GitHub: constr(min_length=1)
    new_dataset: conint(ge=1) | None = Field(alias="New dataset", default=None)
    new_task: conint(ge=2) | None = Field(alias="New task", default=None)
    dataset_annotations: conint(ge=1) | None = Field(
        alias="Dataset annotations", default=None
    )
    bug_fixes: conint(ge=1) | None = Field(alias="Bug fixes", default=None)
    running_models: conint(ge=1) | None = Field(alias="Running Models", default=None)
    review_pr: conint(ge=2) | None = Field(alias="Review PR", default=None)
    paper_writing: int | None = Field(alias="Paper writing", default=None)
    Ideation: int | None = None
    Coordination: int | None = None


def check_max_points(obj: JsonObject, commit_n: str):
    if obj.new_dataset is not None:
        if obj.new_dataset > 50 and commit_n not in commit_exceptions:
            raise ValueError(f"Commit {commit_n} exceeds max points for new_dataset")


# Function to validate JSONL files in a folder
def validate_jsonl_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(folder_path, filename)
            commit_n = os.path.splitext(filename)[0]
            with open(file_path, encoding="utf-8") as file:
                try:
                    # Read JSONL file
                    reader = Reader(file)
                except Exception:
                    raise Exception("Error reading file:", file_path)
                for line in reader:
                    try:
                        # Validate JSON object against schema
                        x = JsonObject(**line)
                        logging.debug(x)
                        check_max_points(x, commit_n)

                    except ValidationError as e:
                        raise Exception(
                            "Validation Error in file:", file_path, line
                        ) from e


# Main function
def main():
    folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "points")
    validate_jsonl_files(folder_path)


if __name__ == "__main__":
    main()
