from pydantic import BaseModel
from typing import Dict, List

class JSONDecoder(BaseModel):
    vocab: Dict[str, int]
    state: str = "START"

    def get_allowed_tokens(self, generated_ids: List[int]) -> List[int]:
        allowed_ids = []

        if len(generated_ids) == 0:
            for token_str, token_id in self.vocab.items():
                if token_str.startswith("{"):
                    allowed_ids.append(token_id)
            return allowed_ids

        # LATER, ADD RULES FOR OTHER KEYS
        return list(self.vocab.values())

    def apply_mask(self, logits: List[float], allowed_ids: List[int]) -> List[float]:
        masked_logits = []

        allowed_set = set(allowed_ids)

        for i, logit in enumerate(logits):
            if i in allowed_set:
                masked_logits.append(logit)
            else:
                masked_logits.append(float('-inf'))
        return masked_logits
