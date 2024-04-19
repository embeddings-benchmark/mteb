import logging
import os
from typing import Optional

from jsonlines import Reader
from pydantic import BaseModel, ConfigDict, Field, ValidationError, conint, constr


# Define a Pydantic model to represent each JSON object
class JsonObject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    GitHub: constr(min_length=1)
    new_dataset: Optional[conint(ge=2)] = Field(alias="New dataset", default=None)
    new_task: Optional[conint(ge=2)] = Field(alias="New task", default=None)
    dataset_annotations: Optional[conint(ge=1)] = Field(
        alias="Dataset annotations", default=None
    )
    bug_fixes: Optional[conint(ge=1)] = Field(alias="Bug fixes", default=None)
    running_models: Optional[conint(ge=1)] = Field(alias="Running Models", default=None)
    review_pr: Optional[conint(ge=2)] = Field(alias="Review PR", default=None)
    paper_writing: Optional[int] = Field(alias="Paper writing", default=None)
    Ideation: Optional[int] = None
    Coordination: Optional[int] = None


# Function to validate JSONL files in a folder
def validate_jsonl_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    # Read JSONL file
                    reader = Reader(file)
                except Exception as e:
                    raise Exception("Error reading file:", file_path)
                for line in reader:
                    try:
                        # Validate JSON object against schema
                        x = JsonObject(**line)
                        logging.debug(x)
                    except ValidationError as e:
                        raise Exception("Validation Error in file:", file_path, line) from e


# Main function
def main():
    folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "points")
    validate_jsonl_files(folder_path)


if __name__ == "__main__":
    main()
