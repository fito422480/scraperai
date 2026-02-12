import json
import logging
import re
from typing import Optional

from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from scraperai.llm.base import BaseJsonLM, _Dict, _DictOrPydanticClass, BaseVision, BasePythonCodeLM
from scraperai.utils.code import extract_python_code

logger = logging.getLogger('scraperai')


def _parse_json_best_effort(text: str) -> dict:
    candidates = []
    raw = (text or "").strip()
    if raw:
        candidates.append(raw)

    # Common model output format: fenced code blocks.
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend([b.strip() for b in fenced_blocks if b.strip()])

    # Extract likely JSON object spans.
    if "{" in raw and "}" in raw:
        candidates.append(raw[raw.find("{"): raw.rfind("}") + 1].strip())

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch != '{':
            continue
        try:
            obj, _ = decoder.raw_decode(raw[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("Model response does not contain a valid JSON object")


class PythonCodeOpenAI(BasePythonCodeLM):
    latest = 'gpt-4o'

    def __init__(self,
                 openai_api_key: str,
                 openai_organization: str = None,
                 model_name: str = latest,
                 **kwargs):
        self.chat = ChatOpenAI(model=model_name,
                               openai_api_key=openai_api_key,
                               openai_organization=openai_organization,
                               max_retries=3,
                               **kwargs)
        self._total_cost = 0

    @property
    def total_cost(self) -> float:
        return self._total_cost

    def invoke(self, messages: list[BaseMessage]) -> str:
        with get_openai_callback() as cb:
            text = self.chat.invoke(messages).content
            self._total_cost += cb.total_cost
            logger.info(f"Total Tokens: {cb.total_tokens}, Total Cost (USD): ${cb.total_cost}")
        return extract_python_code(text)


class JsonOpenAI(BaseJsonLM):
    latest = 'gpt-4o'

    def __init__(self,
                 openai_api_key: str,
                 openai_organization: str = None,
                 model_name: str = latest,
                 schema: Optional[_DictOrPydanticClass] = None,
                 **kwargs):
        model_kwargs = {"response_format": {"type": "json_object"}}
        self.chat = ChatOpenAI(model=model_name,
                               model_kwargs=model_kwargs,
                               openai_api_key=openai_api_key,
                               openai_organization=openai_organization,
                               max_retries=3,
                               **kwargs)
        self.model_with_structure = None
        if schema:
            self.model_with_structure = self.chat.with_structured_output(schema, method='json_mode')
        self._total_cost = 0

    @property
    def total_cost(self) -> float:
        return self._total_cost

    def _repair_to_json(self, raw_text: str) -> dict:
        repair_messages = [
            SystemMessage(
                content="Convert the user text to a strict JSON object. "
                        "Return ONLY JSON without markdown, comments, or explanations."
            ),
            HumanMessage(content=raw_text)
        ]
        repaired = self.chat.invoke(repair_messages).content
        return _parse_json_best_effort(str(repaired))

    def invoke(self, messages: list[BaseMessage]) -> _Dict:
        with get_openai_callback() as cb:
            if self.model_with_structure:
                response = self.model_with_structure.invoke(messages)
            else:
                text = self.chat.invoke(messages).content
                try:
                    response = _parse_json_best_effort(str(text))
                except ValueError:
                    logger.warning("Primary JSON parse failed; attempting repair pass.")
                    try:
                        response = self._repair_to_json(str(text))
                    except Exception:
                        logger.warning(
                            "JSON repair pass failed. Returning empty object "
                            "so validator/retry logic can continue."
                        )
                        response = {}
            self._total_cost += cb.total_cost
            logger.info(f"Total Tokens: {cb.total_tokens}, Total Cost (USD): ${cb.total_cost:.3f}")

        if isinstance(response, BaseModel):
            return response.model_dump()
        else:
            return response


class VisionOpenAI(BaseVision):
    latest = 'gpt-4o'

    def __init__(self,
                 openai_api_key: str,
                 openai_organization: str = None,
                 model_name: str = latest,
                 **kwargs):
        self._total_cost = 0.0
        self.chat = ChatOpenAI(model=model_name,
                               openai_api_key=openai_api_key,
                               openai_organization=openai_organization,
                               **kwargs)

    @property
    def total_cost(self) -> float:
        return self._total_cost

    def invoke(self, messages: list[BaseMessage]) -> str:
        with get_openai_callback() as cb:
            response = self.chat.invoke(messages).content
            self._total_cost += cb.total_cost
        return response
