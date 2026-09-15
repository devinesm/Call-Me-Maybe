import json
import re
from typing import Dict, List

from pydantic import BaseModel, PrivateAttr

from src.schemas import FunctionDefinition

_NUM_PATTERN = re.compile(
    r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?"
)
_PARTIAL_NUM_PATTERN = re.compile(
    r"^-?$|^-?0\.$|^-?[1-9]\d*\.?\d*$|"
    r"^-?(?:0|[1-9]\d*)(?:\.\d+)?[eE][+-]?\d*$"
)


class JSONDecoder(BaseModel):
    vocab: Dict[str, int]
    functions: List[FunctionDefinition]

    _clean_vocab: Dict[int, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context) -> None:
        for t_str, t_id in self.vocab.items():
            self._clean_vocab[t_id] = t_str.replace("Ġ", " ").replace(" ", " ")

    def get_clean_token(self, token_id: int) -> str:
        return self._clean_vocab.get(token_id, "")

    def get_allowed_tokens(self, generated_text: str) -> List[int]:
        allowed_ids = []

        for token_id, clean_token in self._clean_vocab.items():
            proposed_text = generated_text + clean_token
            if self.is_valid_prefix(proposed_text):
                allowed_ids.append(token_id)

        if not allowed_ids:
            raise ValueError(
                f"Deadlock! {generated_text}"
            )

        return allowed_ids

    def is_complete(self, text: str) -> bool:
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False

    def is_valid_prefix(self, text: str) -> bool:
        base_prefix = '{"name": "'

        if len(text) <= len(base_prefix):
            return base_prefix.startswith(text)
        if not text.startswith(base_prefix):
            return False

        for func in self.functions:
            func_prefix = base_prefix + func.name + '", "parameters": {'

            if len(text) <= len(func_prefix):
                if func_prefix.startswith(text):
                    return True
                continue

            if not text.startswith(func_prefix):
                continue

            params_text = text[len(func_prefix):]
            if self.is_valid_params_prefix(params_text, func):
                return True

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
                    val_str = match.group(0)
                    current_text = current_text[len(val_str):]
                else:
                    if _PARTIAL_NUM_PATTERN.match(current_text):
                        return True
                    return False

            elif param_type == "string":
                if not current_text.startswith('"'):
                    return False
                idx = 1
                escaped = False
                in_string = True
                while idx < len(current_text):
                    if escaped:
                        escaped = False
                    elif current_text[idx] == "\\":
                        escaped = True
                    elif current_text[idx] == '"':
                        in_string = False
                        idx += 1
                        break
                    idx += 1
                if in_string:
                    return True
                current_text = current_text[idx:]

            elif param_type == "boolean":
                if "true".startswith(current_text) or "false".startswith(
                    current_text
                ):
                    return True
                if current_text.startswith("true"):
                    current_text = current_text[4:]
                elif current_text.startswith("false"):
                    current_text = current_text[5:]
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

    def apply_mask(
        self, logits: List[float], allowed_ids: List[int]
    ) -> List[float]:
        masked_logits = [float("-inf")] * len(logits)
        for allowed_id in allowed_ids:
            if allowed_id < len(logits):
                masked_logits[allowed_id] = logits[allowed_id]
        return masked_logits
