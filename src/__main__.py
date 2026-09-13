import argparse
import sys
import json
from src.schemas import FunctionDefinition, PromptInput
from typing import List, Dict
from llm_sdk import Small_LLM_Model

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


def load_vocabulary(vocab_path: str) -> Dict[str, int]:
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load vocabulary from {vocab_path}: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_arguments()

    print(f"Loading definitions from: {args.functions_definition}")
    functions_def = load_and_validate_functions(args.functions_definition)

    print(f"Loading inputs from: {args.input}")
    prompts = load_and_validate_prompts(args.input)

    print(f"Output will be saved to: {args.output}")

    print("\n[INFO] Starting Small_LLM_Model...")
    llm = Small_LLM_Model()

    vocab_path = llm.get_path_to_vocab_file()
    vocab = load_vocabulary(vocab_path)
    print(f"[✔] Vocabulary successfully loaded. Total tokens: {len(vocab)}")

    if prompts:
        first_prompt = prompts[0].prompt
        print(f"\n[INFO] Encoding the prompt: '{first_prompt}'")
        input_ids = llm.encode(first_prompt)
        print(f"[✔] Encoded prompt! First tokens (IDs): {input_ids[:10]}")

    print("\n[✔] Setup and LLM ready to proceed to generation!")


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
