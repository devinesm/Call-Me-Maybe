*This project has been created as part of the 42 curriculum by ipinto-m.*

# Call Me Maybe — Introduction to Function Calling in LLMs

## Description

Call Me Maybe is a function calling system that translates natural language prompts into structured JSON function calls using a small language model (Qwen/Qwen3-0.6B). Instead of relying on the model to spontaneously produce valid JSON, the system uses **constrained decoding** to guarantee that every output is 100% valid, parseable JSON that conforms to a predefined function schema.

Given a prompt like `"What is the sum of 2 and 3?"` and a set of available function definitions, the system outputs:

```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
}
```

The model chooses which function to call and extracts the arguments; the constrained decoder ensures the output structure is always valid.

## Instructions

### Prerequisites

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone <repository-url>
cd Call-Me-Maybe-main
uv sync
```

This installs all dependencies (including the bundled `llm_sdk` workspace package) and downloads the Qwen3-0.6B model on first run.

### Running

Default paths (reads from `data/input/`, writes to `data/output/`):

```bash
uv run python -m src
```

Custom paths:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

### Makefile targets

| Target        | Description                                      |
|---------------|--------------------------------------------------|
| `make install`     | Run `uv sync` to install dependencies       |
| `make run`         | Execute the main program                     |
| `make debug`       | Run with Python's built-in debugger (pdb)    |
| `make lint`        | Run flake8 and mypy with required flags      |
| `make lint-strict` | Run flake8 and mypy in strict mode           |
| `make clean`       | Remove `__pycache__`, `.mypy_cache`, etc.    |

## Algorithm Explanation

### Overview

The system works in three stages: **prompting**, **constrained generation**, and **output assembly**.

### 1. Prompting

For each user prompt, a context string is built that lists the available function names and descriptions, followed by the user's request. This context is tokenized and fed to the LLM as input. The prompt is kept intentionally short (only names and descriptions, not full parameter schemas) to stay within the small model's effective context window.

### 2. Constrained Generation (Token-by-Token)

This is the core of the project. Instead of hoping the model produces valid JSON, we **force** it to by modifying its output probabilities at every generation step.

The generation loop works as follows:

1. The LLM produces logits (probability scores) for every token in its vocabulary.
2. The `JSONDecoder` determines which tokens are **allowed** — meaning appending that token to the text generated so far would still be a valid prefix of a well-formed JSON output matching the expected schema.
3. All disallowed tokens have their logits set to negative infinity.
4. The token with the highest remaining logit is selected.
5. That token is appended to both the input sequence and the generated text, and the loop repeats.

This process guarantees that the output is always valid JSON with the correct structure (`{"name": "...", "parameters": {...}}`), correct keys, and correctly typed values.

### 3. Prefix Validation (`is_valid_prefix`)

At every step, the decoder checks whether a proposed string (current generated text + candidate token) could be the beginning of a valid output. The validation follows the fixed JSON structure:

```
{"name": "<function_name>", "parameters": {<key-value pairs>}}
```

The validator walks through this structure left to right:

- First, it checks the `{"name": "` prefix.
- Then it checks if the function name matches one of the defined functions.
- Then it validates the `", "parameters": {` separator.
- Finally, it validates each parameter key-value pair in order, checking that keys match the function definition and that values conform to the declared types (`number`, `string`, or `boolean`).

### 4. Type-Aware Value Validation

Each parameter type has its own validation logic:

- **Numbers**: Validated against the JSON number grammar using regex (`-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?`). A separate partial-match pattern handles incomplete numbers that are still valid prefixes (e.g., `"-"`, `"3."`, `"1e"`).
- **Strings**: Parsed character by character with proper escape handling (`\"`, `\\`, etc.). A string is considered a valid prefix as long as the closing `"` hasn't been reached yet.
- **Booleans**: Matched against the literals `true` and `false`, including partial prefixes (e.g., `"tr"` is a valid prefix of `"true"`).

### 5. Masking and Selection

The `apply_mask` method creates a logits array filled with negative infinity and only restores the original logit values for allowed token IDs. After masking, `numpy.argmax` selects the highest-scoring valid token. This ensures the model's preferences still influence the output (choosing the most probable valid continuation), while structurally invalid tokens are impossible to select.

## Design Decisions

### Fixed key ordering

JSON objects are unordered by definition, but the decoder enforces a fixed key order (`"name"` first, then `"parameters"`, and parameters in definition order). This simplifies the state machine significantly — instead of tracking which keys have been emitted, the validator walks a linear sequence. The tradeoff is that the model can't choose its own key ordering, but since we're forcing the structure anyway, this has no practical downside.

### Pydantic for all data classes

All input schemas (`FunctionDefinition`, `ParameterInfo`, `PromptInput`) and the `JSONDecoder` itself use Pydantic models. This gives us automatic input validation at load time and type-safe attribute access throughout the code.

### Pre-computed clean vocabulary

The vocabulary mapping (token string → cleaned string) is computed once at initialization in `model_post_init`, avoiding repeated string replacements during generation. The cleaned tokens are stored in an `id → string` dictionary for O(1) lookup.

### Simplified prompt

The prompt sent to the LLM includes only function names and descriptions — not the full parameter schemas. The parameter structure is enforced entirely by the constrained decoder, so the model only needs to understand *which function* to call and *what values* to extract. This keeps the prompt short and within the small model's effective reasoning window.

### Error as signal, not fallback

When no valid tokens are found, the decoder raises a `ValueError` instead of falling back to unconstrained generation. A deadlock means there is a bug in the prefix validator, and silently allowing arbitrary tokens would mask it.

## Performance Analysis

### Accuracy

With constrained decoding, every output is guaranteed to be syntactically valid JSON matching the schema. The model's job is reduced to two tasks: selecting the correct function and extracting the right argument values. For straightforward prompts ("What is the sum of 2 and 3?") with a small set of distinct functions, accuracy is expected to be 90%+ on function selection.

### Speed

The main bottleneck is `get_allowed_tokens`, which iterates over the full vocabulary (~150k tokens for Qwen3) at every generation step. Each iteration involves a string concatenation and a call to `is_valid_prefix`. With ~30–50 tokens generated per prompt and ~10 prompts, total prefix checks are in the tens of millions. On a machine with GPU acceleration, the LLM inference itself is fast; the constrained decoding overhead is the limiting factor.

Possible optimizations (not yet implemented):
- **Character-level filtering**: At each step, compute the set of valid next characters from the decoder state, then only check tokens whose first character is in that set.
- **Trie-based vocabulary index**: Build a trie from cleaned token strings to quickly find all tokens matching a given prefix.
- **Early termination in prefix check**: Once the proposed text diverges from all possible valid continuations, reject immediately without checking further functions.

### Reliability

The constrained decoder guarantees that all outputs are parseable JSON. If the model cannot produce a valid output (deadlock), the error is caught and reported explicitly rather than producing garbage.

## Challenges Faced

### Tokenizer encoding

Different models use different conventions for representing whitespace in their vocabulary files. GPT-2-style tokenizers use `Ġ` (U+0120) to represent a leading space, while other tokenizers may use different encodings or raw characters. Getting this mapping right is essential — if spaces are not correctly decoded from token strings, the generated JSON will have missing or extra spaces and fail to parse. The solution was to inspect the actual vocabulary file and handle the relevant replacement patterns.

### Number validation

JSON numbers have a specific grammar that is more restrictive than what most programming languages accept. For example, `03` is not valid JSON (leading zeros are forbidden except for `0` itself), and `-` alone is not a complete number. The challenge was writing a regex that accepts valid *prefixes* of JSON numbers (since numbers are built token by token) without accepting strings that can never become valid numbers.

### Balancing model freedom and structural constraints

The constrained decoder restricts the model heavily, but the model still needs enough freedom to generate semantically correct output — choosing the right function and extracting the right values from the prompt. Finding the right prompt format and the right level of structural constraint required iteration.

### Performance with large vocabularies

Iterating over 150k+ tokens at every generation step is inherently expensive. Keeping the prefix validation fast (avoiding unnecessary allocations, using precompiled regex, pre-computing the clean vocabulary) was important to stay within reasonable execution times.

## Testing Strategy

### Manual validation

Run the program with the provided test prompts and verify:
- The output file is valid JSON.
- Every entry has exactly three keys: `prompt`, `name`, `parameters`.
- Function names match entries in `functions_definition.json`.
- Parameter keys and types match the function definitions.

### Edge cases to test

- **Empty strings**: Prompts that require passing `""` as an argument.
- **Large numbers**: Values like `999999` or `0.0001`.
- **Special characters in strings**: Quotes, backslashes, Unicode.
- **Ambiguous prompts**: Prompts that could match multiple functions.
- **Missing or malformed input files**: Invalid JSON, missing files, empty arrays.
- **Functions with no parameters**: An empty `parameters` object should produce `{}`.
- **Different function sets**: The solution should not be hardcoded to the provided examples.

### Lint and type checking

```bash
make lint          # flake8 + mypy with required flags
make lint-strict   # flake8 + mypy --strict
```

Both should pass without errors.

## Example Usage

### Basic run

```bash
$ uv run python -m src

[INFO] Starting Small_LLM_Model...
[✔] Vocabulary successfully loaded. Total tokens: 151936

[INFO] Starting generation for 11 prompts...

[1/11] Resolving: 'What is the sum of 2 and 3?'
[+] Generated: {"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}

[2/11] Resolving: 'Greet shrek'
[+] Generated: {"name": "fn_greet", "parameters": {"name": "shrek"}}

[3/11] Resolving: 'Reverse the string 'hello''
[+] Generated: {"name": "fn_reverse_string", "parameters": {"s": "hello"}}
...

[✔] Output successfully saved to: data/output/function_calling_results.json
```

### Output format

```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {"a": 2.0, "b": 3.0}
    },
    {
        "prompt": "Greet shrek",
        "name": "fn_greet",
        "parameters": {"name": "shrek"}
    },
    {
        "prompt": "Reverse the string 'hello'",
        "name": "fn_reverse_string",
        "parameters": {"s": "hello"}
    }
]
```

### Custom input files

```bash
uv run python -m src \
    --functions_definition my_functions.json \
    --input my_prompts.json \
    --output results.json
```

## Resources

- [Qwen3-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B) — the default model used in this project
- [Constrained Decoding for LLMs (Outlines paper)](https://arxiv.org/abs/2307.09702) — the foundational paper on using finite-state machines to constrain LLM generation
- [JSON specification (RFC 8259)](https://datatracker.ietf.org/doc/html/rfc8259) — formal grammar for JSON, used to build the number and string validators
- [Pydantic documentation](https://docs.pydantic.dev/) — used for input validation and data modeling
- [uv documentation](https://docs.astral.sh/uv/) — the package manager used for dependency management

### AI usage

AI tools were used during development for:
- **Code review and debugging**: Identifying edge cases in the constrained decoder logic (escape handling, number grammar).
- **Documentation**: Drafting and structuring this README.
- **Research**: Understanding constrained decoding techniques and JSON grammar rules.