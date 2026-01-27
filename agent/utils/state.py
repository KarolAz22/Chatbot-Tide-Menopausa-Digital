from typing import Annotated, Optional, Dict, Any
from typing_extensions import TypedDict
import operator
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage


class StateSchema(TypedDict):

    messages: Annotated[list[AnyMessage], operator.add]
    route: Optional[str]

    user_data: Optional[Dict[str, Any]]
    debug: Optional[Any]
    confirmation: Optional[bool]
    
    # Campos para avaliação e reformulação de respostas
    pass_evaluation: Optional[bool]
    problem: Optional[str]
    



    