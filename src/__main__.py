import argparse
import sys

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


def main() -> None:
    args = parse_arguments()

    print(f"Loading definitions from: {args.functions_definition}")
    # functions_def = load_json_file(args.functions_definition)

    print(f"Loading inputs from: {args.input}")
    # prompts = load_json_file(args.input)

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
