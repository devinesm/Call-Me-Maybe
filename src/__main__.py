import argparse
import sys
import json
from src.schemas import FunctionDefinition, PromptInput
from typing import List

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call Me Maybe - LLM Function Calling Tool")
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the functions definitions file"
    )

    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts file"
    )

    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to save the generated JSON array"
    )
    return parser.parse_args()


def load_and_validate_functions(filepath: str) -> List[FunctionDefinition]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [FunctionDefinition(**item) for item in data]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[ERROR] Failed to read or parse functions file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Validation error in functions file: {e}", file=sys.stderr)
        sys.exit(1)


def load_and_validate_prompts(filepath: str) -> List[PromptInput]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [PromptInput(**item) for item in data]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[ERROR] Failed to read or parse prompts file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Validation error in prompts file: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_arguments()

    print(f"Loading definitions from: {args.functions_definition}")
    functions_def = load_and_validate_functions(args.functions_definition)

    print(f"Loading inputs from: {args.input}")
    prompts = load_and_validate_prompts(args.input)

    print(f"Output will be saved to: {args.output}")

    print("\n[✔] Argument parsing and file loading setup is ready!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WARNING] Process interrupted by user.", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Critical failure in execution -> {e}",
              file=sys.stderr)
        sys.exit(1)
