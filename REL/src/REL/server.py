from REL.response_handler import ResponseHandler

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic import Field
from typing import List, Optional, Literal, Union, Annotated, Tuple

DEBUG = False

app = FastAPI()


Span = Tuple[int, int]


class NamedEntityConfig(BaseModel):
    """Config for named entity linking. For more information, see
    <https://rel.readthedocs.io/en/latest/tutorials/e2e_entity_linking/>
    """

    text: str = Field(..., description="Text for entity linking or disambiguation.")
    spans: Optional[List[Span]] = Field(
        None,
        description=(
            """
For EL: the spans field needs to be set to an empty list. 

For ED: spans should consist of a list of tuples, where each tuple refers to 
the start position and length of a mention.

This is used when mentions are already identified and disambiguation is only 
needed. Each tuple represents start position and length of mention (in 
characters); e.g.,  `[(0, 8), (15,11)]` for mentions 'Nijmegen' and 
'Netherlands' in text 'Nijmegen is in the Netherlands'.
"""
        ),
    )
    tagger: Literal[
        "ner-fast",
        "ner-fast-with-lowercase",
    ] = Field("ner-fast", description="NER tagger to use.")

    class Config:
        schema_extra = {
            "example": {
                "text": "If you're going to try, go all the way - Charles Bukowski.",
                "spans": [(41, 16)],
                "tagger": "ner-fast",
            }
        }

    def response(self):
        """Return response for request."""
        handler = handlers[self.tagger]
        response = handler.generate_response(text=self.text, spans=self.spans)
        return response


class NamedEntityConceptConfig(BaseModel):
    """Config for named entity linking. Not yet implemented."""

    def response(self):
        """Return response for request."""
        response = JSONResponse(
            content={"msg": "Mode `ne_concept` has not been implemeted."},
            status_code=501,
        )
        return response


class ConversationTurn(BaseModel):
    """Specify turns in a conversation. Each turn has a `speaker`
    and an `utterance`."""

    speaker: Literal["USER", "SYSTEM"] = Field(
        ..., description="Speaker for this turn, must be one of `USER` or `SYSTEM`."
    )
    utterance: str = Field(..., description="Input utterance to be annotated.")

    class Config:
        schema_extra = {
            "example": {
                "speaker": "USER",
                "utterance": "I am allergic to tomatoes but we have a lot of famous Italian restaurants here in London.",
            }
        }


class ConversationConfig(BaseModel):
    """Config for conversational entity linking. For more information:
    <https://rel.readthedocs.io/en/latest/tutorials/conversations/>.
    """

    text: List[ConversationTurn] = Field(
        ..., description="Conversation as list of turns between two speakers."
    )
    tagger: Literal[
        "default",
    ] = Field("default", description="NER tagger to use.")

    class Config:
        schema_extra = {
            "example": {
                "text": (
                    {
                        "speaker": "USER",
                        "utterance": "I am allergic to tomatoes but we have a lot of famous Italian restaurants here in London.",
                    },
                    {
                        "speaker": "SYSTEM",
                        "utterance": "Some people are allergic to histamine in tomatoes.",
                    },
                    {
                        "speaker": "USER",
                        "utterance": "Talking of food, can you recommend me a restaurant in my city for our anniversary?",
                    },
                ),
                "tagger": "default",
            }
        }

    def response(self):
        """Return response for request."""
        text = self.dict()["text"]
        conv_handler = conv_handlers[self.tagger]
        response = conv_handler.annotate(text)
        return response


class TurnAnnotation(BaseModel):
    __root__: List[Union[int, str]] = Field(
        ...,
        min_items=4,
        max_items=4,
        description="""
The 4 values of the annotation represent the start index of the word, 
length of the word, the annotated word, and the prediction.
""",
    )

    class Config:
        schema_extra = {"example": [82, 6, "London", "London"]}


class SystemResponse(ConversationTurn):
    """Return input when the speaker equals 'SYSTEM'."""

    speaker: str = "SYSTEM"

    class Config:
        schema_extra = {
            "example": {
                "speaker": "SYSTEM",
                "utterance": "Some people are allergic to histamine in tomatoes.",
            },
        }


class UserResponse(ConversationTurn):
    """Return annotations when the speaker equals 'USER'."""

    speaker: str = "USER"
    annotations: List[TurnAnnotation] = Field(..., description="List of annotations.")

    class Config:
        schema_extra = {
            "example": {
                "speaker": "USER",
                "utterance": "I am allergic to tomatoes but we have a lot of famous Italian restaurants here in London.",
                "annotations": [
                    [17, 8, "tomatoes", "Tomato"],
                    [54, 19, "Italian restaurants", "Italian_cuisine"],
                    [82, 6, "London", "London"],
                ],
            },
        }


TurnResponse = Union[UserResponse, SystemResponse]


class NEAnnotation(BaseModel):
    """Annotation for named entity linking."""

    __root__: List[Union[int, str, float]] = Field(
        ...,
        min_items=7,
        max_items=7,
        description="""
The 7 values of the annotation represent the 
start index, end index, the annotated word, prediction, ED confidence, MD confidence, and tag.
""",
    )

    class Config:
        schema_extra = {
            "example": [41, 16, "Charles Bukowski", "Charles_Bukowski", 0, 0, "NULL"]
        }


class StatusResponse(BaseModel):
    schemaVersion: int
    label: str
    message: str
    color: str


@app.get("/", response_model=StatusResponse)
def server_status():
    """Returns server status."""
    return {
        "schemaVersion": 1,
        "label": "status",
        "message": "up",
        "color": "green",
    }


@app.post("/", response_model=List[NEAnnotation])
@app.post("/ne", response_model=List[NEAnnotation])
def named_entity_linking(config: NamedEntityConfig):
    """Submit your text here for entity disambiguation or linking.

    The REL annotation mode can be selected by changing the path.
    use `/` or `/ne/` for annotating regular text with named
    entities (default), `/ne_concept/` for regular text with both concepts and
    named entities, and `/conv/` for conversations with both concepts and
    named entities.
    """
    if DEBUG:
        return []
    return config.response()


@app.post("/conv", response_model=List[TurnResponse])
def conversational_entity_linking(config: ConversationConfig):
    """Submit your text here for conversational entity linking."""
    if DEBUG:
        return []
    return config.response()


@app.post("/ne_concept", response_model=List[NEAnnotation])
def conceptual_named_entity_linking(config: NamedEntityConceptConfig):
    """Submit your text here for conceptual entity disambiguation or linking."""
    if DEBUG:
        return []
    return config.response()


if __name__ == "__main__":
    import argparse
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("base_url")
    p.add_argument("wiki_version")
    p.add_argument("--ed-model", default="ed-wiki-2019")
    p.add_argument("--ner-model", default="ner-fast", nargs="+")
    p.add_argument("--bind", "-b", metavar="ADDRESS", default="0.0.0.0")
    p.add_argument("--port", "-p", default=5555, type=int)
    args = p.parse_args()

    if not DEBUG:
        from REL.crel.conv_el import ConvEL
        from REL.entity_disambiguation import EntityDisambiguation
        from REL.ner import load_flair_ner

        ed_model = EntityDisambiguation(
            args.base_url,
            args.wiki_version,
            {"mode": "eval", "model_path": args.ed_model},
        )

        handlers = {}

        for ner_model_name in args.ner_model:
            print("Loading NER model:", ner_model_name)
            ner_model = load_flair_ner(ner_model_name)
            handler = ResponseHandler(
                args.base_url, args.wiki_version, ed_model, ner_model
            )
            handlers[ner_model_name] = handler

        conv_handlers = {
            "default": ConvEL(args.base_url, args.wiki_version, ed_model=ed_model)
        }

    uvicorn.run(app, port=args.port, host=args.bind)
