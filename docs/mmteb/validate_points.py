import os

from jsonlines import Reader
from pydantic import BaseModel, ValidationError, conint, constr, field_validator


# Define a Pydantic model to represent each JSON object
class JsonObject(BaseModel):
    GitHub: constr(min_length=1)
    New_dataset: conint(ge=2, le=6) = None
    New_task: conint(ge=2) = None
    Dataset_annotations: conint(ge=1) = None
    Bug_fixes: conint(ge=2, le=10) = None
    Running_Models: conint(ge=1) = None
    Review_PR: conint(ge=2) = None
    Paper_Writing: int = None
    Ideation: int = None
    Coordination: int = None

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
                            JsonObject(**line)
                        except ValidationError as e:
                            print("Validation Error in file:", file_path, e)
                except Exception as e:
                    print("Error reading file:", file_path, e)


# Main function
def main():
    folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "points")
    validate_jsonl_files(folder_path)


if __name__ == "__main__":
    main()
