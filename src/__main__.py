import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

from llm_sdk import Small_LLM_Model
from src.constrained import JSONDecoder
from src.schemas import FunctionDefinition, PromptInput


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - LLM Function Calling"
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        type=str,
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        type=str,
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        type=str,
    )
    return parser.parse_args()


def load_and_validate_functions(filepath: str) -> List[FunctionDefinition]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return [FunctionDefinition(**item) for item in json.load(f)]
    except Exception as exc:
        print(f"[ERROR] Failed to load functions: {exc}", file=sys.stderr)
        sys.exit(1)


def load_and_validate_prompts(filepath: str) -> List[PromptInput]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return [PromptInput(**item) for item in json.load(f)]
    except Exception as exc:
        print(f"[ERROR] Failed to load prompts: {exc}", file=sys.stderr)
        sys.exit(1)


def load_vocabulary(vocab_path: str) -> Dict[str, int]:
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise TypeError("Vocabulary file must contain a JSON object.")
        return {str(key): int(value) for key, value in loaded.items()}
    except Exception as exc:
        print(f"[ERROR] Failed to load vocab: {exc}", file=sys.stderr)
        sys.exit(1)


def build_context_prompt(user_query: str, functions: List[FunctionDefinition]) -> str:
    compact_funcs = [
        {
            "name": f.name,
            "parameters": {k: v.type for k, v in f.parameters.items()}
        }
        for f in functions
    ]
    
    return (
        "You are an AI assistant. Output a JSON object to call a function.\n"
        f"Functions: {json.dumps(compact_funcs)}\n\n"
        f"Request: {user_query}\nJSON:\n"
    )


def main() -> None:
    args = parse_arguments()
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)

    functions_def = load_and_validate_functions(args.functions_definition)
    prompts = load_and_validate_prompts(args.input)

    print("\n[INFO] Starting Small_LLM_Model...")
    llm = Small_LLM_Model()
    vocab = load_vocabulary(llm.get_path_to_vocab_file())

    if prompts:
        decoder = JSONDecoder(vocab=vocab, functions=functions_def)
        results = []

        print(f"\n[INFO] Starting generation for {len(prompts)} prompts...")

        for p_idx, prompt_obj in enumerate(prompts):
            prompt_text = prompt_obj.prompt
            print(f"\n[{p_idx + 1}/{len(prompts)}] Resolving: '{prompt_text}'")

            full_prompt = build_context_prompt(prompt_text, functions_def)
            input_ids = llm.encode(full_prompt)[0].tolist()

            generated_ids: list[int] = []
            generated_text = ""

            while len(generated_ids) < 150:
                logits = llm.get_logits_from_input_ids(input_ids)

                try:
                    allowed_tokens = decoder.get_allowed_tokens(generated_text)
                except ValueError as exc:
                    print(f"\n[ERROR] {exc}")
                    break

                masked_logits = decoder.apply_mask(logits, allowed_tokens)
                next_token_id = int(np.argmax(masked_logits))

                generated_ids.append(next_token_id)
                input_ids.append(next_token_id)

                generated_text += decoder.get_clean_token(next_token_id)
                print(generated_text, end="\r")

                if decoder.is_complete(generated_text):
                    break
            else:
                print(f"\n[WARNING] Prompt {p_idx + 1} hit token limit!")

            print(f"\n[+] Generated: {generated_text}")

            try:
                parsed_json = json.loads(generated_text)
                results.append(
                    {
                        "prompt": prompt_text,
                        "name": parsed_json["name"],
                        "parameters": parsed_json["parameters"],
                    }
                )
            except json.JSONDecodeError:
                pass

        with open(args.output, "w", encoding="utf-8") as out_f:
            json.dump(results, out_f, indent=4)
        print(f"\n[✔] Output successfully saved to: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WARNING] Interrupted by user.", file=sys.stderr)
        sys.exit(0)
    except Exception as exc:
        print(
            f"\n[ERROR] Critical execution failure -> {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
