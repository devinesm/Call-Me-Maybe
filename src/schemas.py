from typing import Dict, Any

from pydantic import BaseModel


class ParameterInfo(BaseModel):
    type: Any


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterInfo]
    returns: ParameterInfo


class PromptInput(BaseModel):
    prompt: str
