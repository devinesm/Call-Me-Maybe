from pydantic import BaseModel
from typing import Dict, List
from src.schemas import FunctionDefinition

class JSONDecoder(BaseModel):
    vocab: Dict[str, int]
    functions: List[FunctionDefinition]

    def get_allowed_tokens(self, generated_text: str) -> List[int]:
        allowed_ids = []

        for token_str, token_id in self.vocab.items():
            clean_token = token_str.replace("Ġ", " ")
            proposed_text = generated_text + clean_token

            if self.is_valid_prefix(proposed_text):
                allowed_ids.append(token_id)

        if not allowed_ids:
            return list(self.vocab.values())

        return allowed_ids

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

    def is_valid_params_prefix(self, params_text: str, func: FunctionDefinition) -> bool:
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

            if current_text == "": return True
            if len(current_text) <= len(key_prefix):
                return key_prefix.startswith(current_text)
            if not current_text.startswith(key_prefix):
                return False

            current_text = current_text[len(key_prefix):]
            if current_text == "": return True

            if param_type == "number":
                val_str = ""
                for char in current_text:
                    if char in "0123456789.-":
                        val_str += char
                    else:
                        break
                if len(val_str) == 0 and current_text[0] not in "0123456789.-":
                    return False
                current_text = current_text[len(val_str):]

            elif param_type == "string":
                if not current_text.startswith('"'):
                    return False
                end_quote_idx = current_text.find('"', 1)
                if end_quote_idx == -1:
                    return True
                current_text = current_text[end_quote_idx + 1:]

            elif param_type == "boolean":
                if "true".startswith(current_text) or "false".startswith(current_text):
                    return True
                if current_text.startswith("true"):
                    current_text = current_text[4:]
                elif current_text.startswith("false"):
                    current_text = current_text[5:]
                else:
                    return False

            if current_text == "": return True

            suffix = ", " if i < len(keys) - 1 else "}}"

            if len(current_text) <= len(suffix):
                return suffix.startswith(current_text)
            if not current_text.startswith(suffix):
                return False

            current_text = current_text[len(suffix):]

        return current_text == ""

    def apply_mask(self, logits: List[float], allowed_ids: List[int]) -> List[float]:
        masked_logits = []
        allowed_set = set(allowed_ids)

        for i, logit in enumerate(logits):
            if i in allowed_set:
                masked_logits.append(logit)
            else:
                masked_logits.append(float('-inf'))
        return masked_logits
