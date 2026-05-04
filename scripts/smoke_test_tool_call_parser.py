from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tec_agents.llm.tool_call_parser import (
    ParseErrorCode,
    parse_model_output,
    parse_tool_call,
    parse_final_answer,
    extract_json_block,
)


def print_case(title: str, text: str) -> None:
    print("=" * 100)
    print(title)
    print("-" * 100)

    result = parse_model_output(text)

    print("tool_call:", result.tool_call)
    print("final_answer:", result.final_answer)
    print("error_code:", result.error_code)
    print("error_message:", result.error_message)


def test_valid_tool_call() -> None:
    text = """
<tool_call>
{"name": "tec_get_timeseries", "arguments": {"dataset_ref": "default", "region_id": "midlat_europe", "start": "2024-03-01", "end": "2024-04-01"}}
</tool_call>
"""

    result = parse_tool_call(text)

    assert result.tool_call is not None
    assert result.tool_call.name == "tec_get_timeseries"
    assert result.tool_call.arguments["dataset_ref"] == "default"
    assert result.tool_call.arguments["region_id"] == "midlat_europe"
    assert result.error_code is None


def test_valid_final_answer() -> None:
    text = """
<final_answer>
High TEC intervals were detected for midlat_europe in March 2024.
</final_answer>
"""

    answer = parse_final_answer(text)

    assert answer is not None
    assert "midlat_europe" in answer


def test_invalid_json() -> None:
    text = """
<tool_call>
{"name": "tec_get_timeseries", "arguments": {"dataset_ref": "default",}}
</tool_call>
"""

    result = parse_tool_call(text)

    assert result.tool_call is None
    assert result.error_code == ParseErrorCode.INVALID_JSON


def test_missing_tool_name() -> None:
    text = """
<tool_call>
{"arguments": {"dataset_ref": "default"}}
</tool_call>
"""

    result = parse_tool_call(text)

    assert result.tool_call is None
    assert result.error_code == ParseErrorCode.MISSING_TOOL_NAME


def test_missing_arguments() -> None:
    text = """
<tool_call>
{"name": "tec_get_timeseries"}
</tool_call>
"""

    result = parse_tool_call(text)

    assert result.tool_call is None
    assert result.error_code == ParseErrorCode.MISSING_ARGUMENTS


def test_unknown_format() -> None:
    text = "I think we should call a tool, but I forgot the format."

    result = parse_model_output(text)

    assert result.tool_call is None
    assert result.final_answer is None
    assert result.error_code == ParseErrorCode.UNKNOWN_FORMAT


def test_extract_json_with_surrounding_text() -> None:
    text = """
Some extra model text.
{"name": "tec_compare_regions", "arguments": {"dataset_ref": "default", "regions": ["midlat_europe", "highlat_north"]}}
Some extra text.
"""

    data = extract_json_block(text)

    assert data["name"] == "tec_compare_regions"
    assert data["arguments"]["dataset_ref"] == "default"


def main() -> None:
    examples = [
        (
            "VALID TOOL CALL",
            """
<tool_call>
{"name": "tec_get_timeseries", "arguments": {"dataset_ref": "default", "region_id": "midlat_europe", "start": "2024-03-01", "end": "2024-04-01"}}
</tool_call>
""",
        ),
        (
            "VALID FINAL ANSWER",
            """
<final_answer>
High TEC intervals were detected for midlat_europe in March 2024.
</final_answer>
""",
        ),
        (
            "BROKEN JSON",
            """
<tool_call>
{"name": "tec_get_timeseries", "arguments": {"dataset_ref": "default",}}
</tool_call>
""",
        ),
        (
            "UNKNOWN FORMAT",
            "Call tec_get_timeseries for Europe in March.",
        ),
    ]

    for title, text in examples:
        print_case(title, text)

    test_valid_tool_call()
    test_valid_final_answer()
    test_invalid_json()
    test_missing_tool_name()
    test_missing_arguments()
    test_unknown_format()
    test_extract_json_with_surrounding_text()

    print("=" * 100)
    print("All parser smoke tests passed.")


if __name__ == "__main__":
    main()