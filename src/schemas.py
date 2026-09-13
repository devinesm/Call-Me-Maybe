from typing import Dict
from pydantic import BaseModel

class ParameterInfo(BaseModel):
    type: str

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterInfo]
    returns: ParameterInfo

class PromptInput(BaseModel):
    prompt: str
