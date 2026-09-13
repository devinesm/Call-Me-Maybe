import argparse
import sys
import json
from src.schemas import FunctionDefinition, PromptInput
from typing import List, Dict
from llm_sdk import Small_LLM_Model
from src.constrained import JSONDecoder
import numpy as np

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


def build_context_prompt(user_query: str, functions: List[FunctionDefinition]) -> str:
    funcs_json = json.dumps([f.model_dump() for f in functions], indent=2)

    prompt = (
        "You are an AI assistant. Your task is to output a JSON object to call a function.\n"
        f"Available functions:\n{funcs_json}\n\n"
        f"User request: {user_query}\n"
        "Output the exact JSON function call.\n"
        "JSON:\n"
    )
    return prompt


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
        decoder = JSONDecoder(vocab=vocab, functions=functions_def)
        results = []
        id_to_str = {v: k for k, v in vocab.items()}

        print(f"\n[INFO] Starting automation for {len(prompts)} prompts...")

        for p_idx, prompt_obj in enumerate(prompts):
            prompt_text = prompt_obj.prompt
            print(f"\n[{p_idx+1}/{len(prompts)}] Resolving: '{prompt_text}'")

            full_prompt = build_context_prompt(prompt_text, functions_def)
            tensor_ids = llm.encode(full_prompt)
            input_ids = tensor_ids[0].tolist()

            generated_ids = []
            generated_text = ""

            while len(generated_ids) < 100:
                logits = llm.get_logits_from_input_ids(input_ids)
                allowed_tokens = decoder.get_allowed_tokens(generated_text)
                masked_logits = decoder.apply_mask(logits, allowed_tokens)

                next_token_id = int(np.argmax(masked_logits))

                generated_ids.append(next_token_id)
                input_ids.append(next_token_id)

                token_str = id_to_str[next_token_id]
                clean_token = token_str.replace("Ġ", " ")
                generated_text += clean_token

                print(generated_text, end="\r")

                if generated_text.endswith("}}"):
                    break

            print(f"\n[+] Generated successfully: {generated_text}")

            try:
                parsed_json = json.loads(generated_text)
                results.append({
                    "prompt": prompt_text,
                    "name": parsed_json["name"],
                    "parameters": parsed_json["parameters"]
                })
            except json.JSONDecodeError:
                print(f"[WARNING] Model failed to generate valid JSON for prompt {p_idx+1}")

        with open(args.output, 'w', encoding='utf-8') as out_f:
            json.dump(results, out_f, indent=4)
        print(f"\n[✔] All prompts processed! Output saved to: {args.output}")


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
