from evo_prompt.comfy_wrapper import _replace_in_json

def test_replace_in_json_strings():
    """Test replacing strings in nested dicts and lists."""
    data = {
        "text": "hello",
        "list": ["hello", "world"],
        "nested": {
            "key": "hello",
            "deep": [{"item": "hello"}]
        }
    }
    expected = {
        "text": "hi",
        "list": ["hi", "world"],
        "nested": {
            "key": "hi",
            "deep": [{"item": "hi"}]
        }
    }
    assert _replace_in_json(data, "hello", "hi") == expected

def test_replace_in_json_integers():
    """Test replacing integers (exact match)."""
    data = {
        "id": 777,
        "values": [777, 123],
        "nested": {"id": 777}
    }
    expected = {
        "id": 999,
        "values": [999, 123],
        "nested": {"id": 999}
    }
    assert _replace_in_json(data, 777, 999) == expected

def test_replace_in_json_no_match():
    """Test that it returns the original object if no match is found."""
    data = {"a": "hello"}
    assert _replace_in_json(data, "missing", "new") == data
