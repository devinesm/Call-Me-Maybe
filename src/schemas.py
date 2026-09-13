from typing import Dict, Literal

from pydantic import BaseModel


class ParameterInfo(BaseModel):
    type: Literal["number", "string", "boolean"]


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterInfo]
    returns: ParameterInfo


class PromptInput(BaseModel):
    prompt: str
