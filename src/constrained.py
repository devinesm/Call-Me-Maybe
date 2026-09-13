from pydantic import BaseModel
from typing import Dict, List

class JSONDecoder(BaseModel):
    vocab: Dict[str, int]
    valid_functions: List[str] = []

    def get_allowed_tokens(self, generated_text: str) -> List[int]:
        allowed_ids = []
        target_start = '{"name": "'

        for token_str, token_id in self.vocab.items():
            clean_token = token_str.replace("Ġ", " ")
            proposed_text = generated_text + clean_token

            if len(proposed_text) <= len(target_start):
                if target_start.startswith(proposed_text):
                    allowed_ids.append(token_id)
                continue

            if not proposed_text.startswith(target_start):
                continue

            content_after_start = proposed_text[len(target_start):]

            is_valid_path = False
            for func_name in self.valid_functions:
                if func_name.startswith(content_after_start):
                    is_valid_path = True
                    break
                if content_after_start.startswith(func_name + '"'):
                    is_valid_path = True
                    break

            if is_valid_path:
                allowed_ids.append(token_id)

        if not allowed_ids:
            return list(self.vocab.values())

        return allowed_ids

    def apply_mask(self, logits: List[float], allowed_ids: List[int]) -> List[float]:
        masked_logits = []
        allowed_set = set(allowed_ids)

        for i, logit in enumerate(logits):
            if i in allowed_set:
                masked_logits.append(logit)
            else:
                masked_logits.append(float('-inf'))
        return masked_logits
