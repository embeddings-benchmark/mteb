import logging
import os
from typing import Optional

from jsonlines import Reader
from pydantic import BaseModel, Field, ValidationError, conint, constr, field_validator


# Define a Pydantic model to represent each JSON object
class JsonObject(BaseModel):
    GitHub: constr(min_length=1)
    new_dataset: Optional[conint(ge=2)] = Field(alias="New dataset", default=None)
    new_task: Optional[conint(ge=2)] = Field(alias="New task", default=None)
    dataset_annotations: Optional[conint(ge=1)] = Field(
        alias="Dataset annotations", default=None
    )
    bug_fixes: Optional[conint(ge=2)] = Field(alias="Dataset annotations", default=None)
    running_models: Optional[conint(ge=1)] = Field(alias="Running models", default=None)
    review_pr: Optional[conint(ge=2)] = Field(alias="Review PR", default=None)
    paper_writing: Optional[int] = Field(alias="Paper writing", default=None)
    ideation: Optional[int] = None
    coordination: Optional[int] = None

    @field_validator("*")
    def check_optional_fields(cls, value):
        if value == "":
            raise ValueError("Optional fields cannot be empty.")
        return value


# Function to validate JSONL files in a folder
def validate_jsonl_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    # Read JSONL file
                    reader = Reader(file)
                    for line in reader:
                        try:
                            # Validate JSON object against schema
                            x = JsonObject(**line)
                            logging.debug(x)
                        except ValidationError as e:
                            print("Validation Error in file:", file_path, e, line)
                except Exception as e:
                    print("Error reading file:", file_path, e)


# Main function
def main():
    folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "points")
    validate_jsonl_files(folder_path)


if __name__ == "__main__":
    main()
