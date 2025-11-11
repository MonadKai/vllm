from typing import Union, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class TruncationDirection(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class Sequence(BaseModel):
    @classmethod
    def validate_sequence(cls, v):
        if isinstance(v, str):
            return {"type": "single", "value": v}
        elif isinstance(v, list):
            if len(v) == 1:
                return {"type": "single", "value": v[0]}
            elif len(v) == 2:
                return {"type": "pair", "first": v[0], "second": v[1]}
            else:
                raise ValueError(
                    "sequence must be a single string or a list of two strings"
                )
        else:
            raise ValueError("input must be a string or a list of strings")

    def count_chars(self) -> int:
        if hasattr(self, "value"):
            return len(self.value)
        else:
            return len(self.first) + len(self.second)


class PredictInput(BaseModel):
    @classmethod
    def validate_predict_input(cls, v):
        if isinstance(v, str):
            return {
                "type": "single",
                "sequence": {"type": "single", "value": v},
            }
        elif isinstance(v, list):
            if len(v) == 0:
                raise ValueError("input cannot be empty")

            first = v[0]
            if isinstance(first, str):
                if len(v) == 1:
                    return {
                        "type": "single",
                        "sequence": {"type": "single", "value": first},
                    }
                elif len(v) == 2:
                    return {
                        "type": "single",
                        "sequence": {
                            "type": "pair",
                            "first": first,
                            "second": v[1],
                        },
                    }
                else:
                    raise ValueError(
                        "a single sequence can only have two elements"
                    )
            elif isinstance(first, list):
                sequences = []
                for item in v:
                    if isinstance(item, str):
                        sequences.append({"type": "single", "value": item})
                    elif isinstance(item, list):
                        if len(item) == 1:
                            sequences.append(
                                {"type": "single", "value": item[0]}
                            )
                        elif len(item) == 2:
                            sequences.append(
                                {
                                    "type": "pair",
                                    "first": item[0],
                                    "second": item[1],
                                }
                            )
                        else:
                            raise ValueError(
                                "each sequence must be a single string or a list of two strings"
                            )
                    else:
                        raise ValueError(
                            "each element in the batch must be a string or a list of strings"
                        )
                return {"type": "batch", "sequences": sequences}
            else:
                raise ValueError(
                    "the first element in the batch must be a string or a list of strings"
                )
        else:
            raise ValueError("input must be a string or a list of strings")


class PredictRequest(BaseModel):
    inputs: PredictInput = Field(..., description="model inputs")
    truncate: Optional[bool] = Field(False, description="whether to truncate")
    truncation_direction: TruncationDirection = Field(
        TruncationDirection.RIGHT, description="truncation direction"
    )

    raw_scores: bool = Field(False, description="whether to return raw scores")

    class Config:
        schema_extra = {
            "example": {
                "inputs": "What is Deep Learning?",
                "truncate": False,
                "truncation_direction": "right",
                "raw_scores": False,
            }
        }


class Prediction(BaseModel):
    score: float = Field(..., description="prediction score", example=0.5)
    label: str = Field(
        ..., description="prediction label", example="admiration"
    )

    class Config:
        schema_extra = {"example": {"score": 0.5, "label": "admiration"}}


class PredictResponse(BaseModel):
    predictions: Union[List[Prediction], List[List[Prediction]]] = Field(
        ..., description="prediction results"
    )


class RerankRequest(BaseModel):
    query: str = Field(
        ..., description="query text", example="What is Deep Learning?"
    )
    texts: List[str] = Field(
        ..., description="texts to rerank", example=["Deep Learning is ..."]
    )
    truncate: Optional[bool] = Field(False, description="whether to truncate")
    truncation_direction: TruncationDirection = Field(
        TruncationDirection.RIGHT, description="truncation direction"
    )
    raw_scores: bool = Field(False, description="whether to return raw scores")
    return_text: bool = Field(False, description="whether to return text")

    class Config:
        schema_extra = {
            "example": {
                "query": "What is Deep Learning?",
                "texts": ["Deep Learning is ..."],
                "truncate": False,
                "truncation_direction": "right",
                "raw_scores": False,
                "return_text": False,
            }
        }


class Rank(BaseModel):
    index: int = Field(..., description="index", example=0)
    text: Optional[str] = Field(
        None, description="text content", example="Deep Learning is ..."
    )
    score: float = Field(..., description="similarity score", example=1.0)

    class Config:
        schema_extra = {
            "example": {
                "index": 0,
                "text": "Deep Learning is ...",
                "score": 1.0,
            }
        }


RerankResponse = list[Rank]


Input = Union[Union[str, list[str]], Union[list[int], list[list[int]]]]


class SimilarityInput(BaseModel):
    source_sentence: str = Field(
        ..., description="source sentence", example="What is Deep Learning?"
    )
    sentences: List[str] = Field(
        ...,
        description="sentences to compare",
        example=["What is Machine Learning?"],
    )

    class Config:
        schema_extra = {
            "example": {
                "source_sentence": "What is Deep Learning?",
                "sentences": ["What is Machine Learning?"],
            }
        }


class SimilarityParameters(BaseModel):
    truncate: Optional[bool] = Field(False, description="whether to truncate")
    truncation_direction: TruncationDirection = Field(
        TruncationDirection.RIGHT, description="truncation direction"
    )
    prompt_name: Optional[str] = Field(None, description="prompt name")

    class Config:
        schema_extra = {
            "example": {
                "truncate": False,
                "truncation_direction": "right",
                "prompt_name": None,
            }
        }


class SimilarityRequest(BaseModel):
    inputs: SimilarityInput = Field(..., description="similarity inputs")
    parameters: Optional[SimilarityParameters] = Field(
        None, description="similarity parameters"
    )

    class Config:
        schema_extra = {
            "example": {
                "inputs": {
                    "source_sentence": "What is Deep Learning?",
                    "sentences": ["What is Machine Learning?"],
                },
                "parameters": {
                    "truncate": False,
                    "truncation_direction": "right",
                    "prompt_name": None,
                },
            }
        }


class SimilarityResponse(BaseModel):
    similarities: List[float] = Field(
        ..., description="similarity scores", example=[0.0, 1.0, 0.5]
    )


class EmbedRequest(BaseModel):
    inputs: Input = Field(..., description="inputs")
    truncate: Optional[bool] = Field(False, description="whether to truncate")
    truncation_direction: TruncationDirection = Field(
        TruncationDirection.RIGHT, description="truncation direction"
    )
    prompt_name: Optional[str] = Field(None, description="prompt name")
    normalize: bool = Field(True, description="whether to normalize")

    class Config:
        schema_extra = {
            "example": {
                "inputs": "What is Deep Learning?",
                "truncate": False,
                "truncation_direction": "right",
                "prompt_name": None,
                "normalize": True,
            }
        }


EmbedResponse = list[list[float]]


class SparseValue(BaseModel):
    index: int = Field(..., description="index")
    value: float = Field(..., description="value")

    class Config:
        schema_extra = {"example": {"index": 0, "value": 1.0}}


class EmbedSparseRequest(BaseModel):
    inputs: Input = Field(..., description="inputs")
    truncate: Optional[bool] = Field(False, description="whether to truncate")
    truncation_direction: TruncationDirection = Field(
        TruncationDirection.RIGHT, description="truncation direction"
    )
    prompt_name: Optional[str] = Field(None, description="prompt name")

    class Config:
        schema_extra = {
            "example": {
                "inputs": "What is Deep Learning?",
                "truncate": False,
                "truncation_direction": "right",
                "prompt_name": None,
            }
        }


class EmbedSparseResponse(BaseModel):
    sparse_embeddings: List[List[SparseValue]] = Field(
        ..., description="sparse embeddings"
    )


class EmbedAllRequest(BaseModel):
    inputs: Input = Field(..., description="inputs")
    truncate: Optional[bool] = Field(False, description="whether to truncate")
    truncation_direction: TruncationDirection = Field(
        TruncationDirection.RIGHT, description="truncation direction"
    )
    prompt_name: Optional[str] = Field(None, description="prompt name")

    class Config:
        schema_extra = {
            "example": {
                "inputs": "What is Deep Learning?",
                "truncate": False,
                "truncation_direction": "right",
                "prompt_name": None,
            }
        }


class EmbedAllResponse(BaseModel):
    all_embeddings: List[List[List[float]]] = Field(
        ..., description="all embeddings", example=[[[0.0, 1.0, 2.0]]]
    )


class TokenizeInput(BaseModel):
    @classmethod
    def validate_tokenize_input(cls, v):
        if isinstance(v, str):
            return {"type": "single", "value": v}
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            return {"type": "batch", "values": v}
        else:
            raise ValueError(
                "tokenize inputs must be a string or a list of strings"
            )


class TokenizeRequest(BaseModel):
    inputs: TokenizeInput = Field(..., description="tokenize inputs")
    add_special_tokens: bool = Field(
        True, description="whether to add special tokens"
    )
    prompt_name: Optional[str] = Field(None, description="prompt name")

    class Config:
        schema_extra = {
            "example": {
                "inputs": "test",
                "add_special_tokens": True,
                "prompt_name": None,
            }
        }


class SimpleToken(BaseModel):
    id: int = Field(..., description="token id", example=0)
    text: str = Field(..., description="token text", example="test")
    special: bool = Field(
        ..., description="whether is special token", example=False
    )
    start: Optional[int] = Field(None, description="start position", example=0)
    stop: Optional[int] = Field(None, description="end position", example=2)

    class Config:
        schema_extra = {
            "example": {
                "id": 0,
                "text": "test",
                "special": False,
                "start": 0,
                "stop": 2,
            }
        }


class TokenizeResponse(BaseModel):
    tokens: List[List[SimpleToken]] = Field(..., description="tokenize results")


class InputIds(BaseModel):
    @classmethod
    def validate_input_ids(cls, v):
        if isinstance(v, list) and all(isinstance(x, int) for x in v):
            return {"type": "single", "ids": v}
        elif isinstance(v, list) and all(
            isinstance(x, list) and all(isinstance(y, int) for y in x)
            for x in v
        ):
            return {"type": "batch", "ids_list": v}
        else:
            raise ValueError(
                "input ids must be a list of integers or a list of lists of integers"
            )


class DecodeRequest(BaseModel):
    ids: InputIds = Field(..., description="input ids")
    skip_special_tokens: bool = Field(
        True, description="whether to skip special tokens"
    )

    class Config:
        schema_extra = {
            "example": {"ids": [0, 1, 2], "skip_special_tokens": True}
        }


class DecodeResponse(BaseModel):
    texts: List[str] = Field(..., description="decode texts", example=["test"])