import pytest
from pydantic import BaseModel, Field
from google.adk.utils._schema_utils import validate_schema

class DummyModel(BaseModel):
    name: str
    age: int

def test_validate_schema_clean_json():
    # Test valid clean JSON
    json_text = '{"name": "Alice", "age": 25}'
    result = validate_schema(DummyModel, json_text)
    assert result == {"name": "Alice", "age": 25}

def test_validate_schema_markdown_code_block():
    # Test JSON inside markdown code blocks
    json_text = """
    ```json
    {"name": "Bob", "age": 30}
    ```
    """
    result = validate_schema(DummyModel, json_text)
    assert result == {"name": "Bob", "age": 30}

def test_validate_schema_conversational_filler():
    # Test JSON nested inside conversational text
    json_text = """
    I can help you with that! Here is the requested structured format:
    {"name": "Charlie", "age": 35}
    Please let me know if you need anything else.
    """
    result = validate_schema(DummyModel, json_text)
    assert result == {"name": "Charlie", "age": 35}

def test_validate_schema_pure_plain_text():
    # Test pure plain text with no JSON - should raise ValueError
    plain_text = "I can help you with this..."
    with pytest.raises(ValueError) as excinfo:
        validate_schema(DummyModel, plain_text)
    assert "Invalid JSON response" in str(excinfo.value)
