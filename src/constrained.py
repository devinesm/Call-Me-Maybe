import json
import re
from typing import Dict, List, Any, cast

import numpy as np
from pydantic import BaseModel, PrivateAttr

from src.schemas import FunctionDefinition

_NUM_PATTERN = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
_PARTIAL_NUM_PATTERN = re.compile(
    r"^-?$|^-?0\.$|^-?[1-9]\d*\.?\d*$|"
    r"^-?(?:0|[1-9]\d*)(?:\.\d+)?[eE][+-]?\d*$"
)
_STRING_PATTERN = re.compile(r'^"(?:[^"\\]|\\.)*"')


class JSONDecoder(BaseModel):
    vocab: Dict[str, int]
    functions: List[FunctionDefinition]

    _clean_vocab: Dict[int, str] = PrivateAttr(default_factory=dict)
    _vocab_items: List[tuple] = PrivateAttr(default_factory=list)
    _tokens_no_quotes: List[int] = PrivateAttr(default_factory=list)
    _tokens_with_quotes: List[tuple] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        for t_str, t_id in self.vocab.items():
            clean = t_str.replace("Ġ", " ").replace(" ", " ")
            self._clean_vocab[t_id] = clean
            self._vocab_items.append((t_id, clean))

            if '"' not in clean and '\\' not in clean and '\n' not in clean:
                self._tokens_no_quotes.append(t_id)
            else:
                self._tokens_with_quotes.append((t_id, clean))

    def get_clean_token(self, token_id: int) -> str:
        return self._clean_vocab.get(token_id, "")

    def is_stuck_in_string(
        self, params_text: str, func: FunctionDefinition
    ) -> bool:
        current_text = params_text
        keys = list(func.parameters.keys())
        for i, key in enumerate(keys):
            param_type = func.parameters[key].type
            key_prefix = f'"{key}": '

            if not current_text.startswith(key_prefix):
                return False
            current_text = current_text[len(key_prefix):]
            if current_text == "":
                return False

            if param_type == "string":
                match = _STRING_PATTERN.match(current_text)
                if match:
                    current_text = current_text[len(match.group(0)):]
                else:
                    return current_text.startswith('"')
            elif param_type == "number":
                match = _NUM_PATTERN.match(current_text)
                if match:
                    current_text = current_text[len(match.group(0)):]
                else:
                    return False
            elif param_type == "boolean":
                if current_text.startswith("true"):
                    current_text = current_text[4:]
                elif current_text.startswith("false"):
                    current_text = current_text[5:]
                else:
                    return False

            suffix = ", " if i < len(keys) - 1 else "}}"
            if not current_text.startswith(suffix):
                return False
            current_text = current_text[len(suffix):]
        return False

    def get_allowed_tokens(self, generated_text: str) -> List[int]:
        allowed_ids = []
        base_prefix = '{"name": "'
        len_gen = len(generated_text)
        len_base = len(base_prefix)

        viable_funcs = []
        in_string_fast = False

        for func in self.functions:
            f_prefix = base_prefix + func.name + '", "parameters": {'
            len_f_prefix = len(f_prefix)
            if len_gen <= len_f_prefix:
                if f_prefix.startswith(generated_text):
                    viable_funcs.append((func, f_prefix, len_f_prefix))
            elif generated_text.startswith(f_prefix):
                in_str = self.is_stuck_in_string(
                    generated_text[len_f_prefix:], func
                )
                if in_str:
                    in_string_fast = True
                viable_funcs.append((func, f_prefix, len_f_prefix))

        is_valid_params = self.is_valid_params_prefix

        if in_string_fast:
            allowed_ids.extend(self._tokens_no_quotes)
            items_to_check = self._tokens_with_quotes
        else:
            items_to_check = self._vocab_items

        for token_id, clean_token in items_to_check:
            proposed = generated_text + clean_token
            len_prop = len_gen + len(clean_token)

            if len_prop <= len_base:
                if base_prefix.startswith(proposed):
                    allowed_ids.append(token_id)
                continue

            if not proposed.startswith(base_prefix):
                continue

            is_valid = False
            for func, f_prefix, len_f_prefix in viable_funcs:
                if len_prop <= len_f_prefix:
                    if f_prefix.startswith(proposed):
                        is_valid = True
                        break
                    continue

                if not proposed.startswith(f_prefix):
                    continue

                if is_valid_params(proposed[len_f_prefix:], func):
                    is_valid = True
                    break

            if is_valid:
                allowed_ids.append(token_id)

        if not allowed_ids:
            raise ValueError(f"Deadlock! {generated_text}")

        return allowed_ids

    def is_complete(self, text: str) -> bool:
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False

    def is_valid_params_prefix(
        self, params_text: str, func: FunctionDefinition
    ) -> bool:
        current_text = params_text
        keys = list(func.parameters.keys())

        if not keys:
            suffix = "}}"
            if len(current_text) <= len(suffix):
                return suffix.startswith(current_text)
            return current_text == suffix

        for i, key in enumerate(keys):
            param_type = func.parameters[key].type
            key_prefix = f'"{key}": '

            if current_text == "":
                return True
            if len(current_text) <= len(key_prefix):
                return key_prefix.startswith(current_text)
            if not current_text.startswith(key_prefix):
                return False

            current_text = current_text[len(key_prefix):]
            if current_text == "":
                return True

            if param_type == "number":
                match = _NUM_PATTERN.match(current_text)
                if match:
                    current_text = current_text[len(match.group(0)):]
                else:
                    return bool(_PARTIAL_NUM_PATTERN.match(current_text))

            elif param_type == "string":
                if not current_text.startswith('"'):
                    return False
                if '{"name"' in current_text:
                    return False

                match = _STRING_PATTERN.match(current_text)
                if match:
                    current_text = current_text[len(match.group(0)):]
                else:
                    return True

            elif param_type == "boolean":
                if "true".startswith(current_text) or \
                   "false".startswith(current_text):
                    current_text = (
                        current_text[4:] if current_text.startswith("true")
                        else current_text[5:]
                    )
                else:
                    return False

            if current_text == "":
                return True

            suffix = ", " if i < len(keys) - 1 else "}}"
            if len(current_text) <= len(suffix):
                return suffix.startswith(current_text)
            if not current_text.startswith(suffix):
                return False

            current_text = current_text[len(suffix):]

        return current_text == ""

    def apply_mask(self,
                   logits: List[float],
                   allowed_ids: List[int]) -> List[float]:
        np_logits = np.array(logits)
        masked = np.full_like(np_logits, -np.inf)

        valid_ids = np.array(allowed_ids, dtype=np.int32)
        valid_ids = valid_ids[valid_ids < len(logits)]

        masked[valid_ids] = np_logits[valid_ids]
        return cast(List[float], masked.tolist())
