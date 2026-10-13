from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import os
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

from PIL import Image

from scanwich.layout import (
    Box,
    FigureLocation,
    LayoutFormatError,
    PageLayout,
    figure_location_from_json,
    page_layout_from_json,
    plain_text,
)
from scanwich.models import OcrRegion, Point

DEFAULT_MODEL = "deepseek/deepseek-v4-flash-vision-exp"
DEFAULT_MODEL_ALIASES = {
    "deepseek": DEFAULT_MODEL,
    "glm-ocr": "zai-org/GLM-OCR",
    "luna": "gpt-5.6-luna",
}
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_TOKENS = 65_536
DEFAULT_REASONING_EFFORT = "low"
# The first request and the requests that ask the model to correct an unusable reply.
DEFAULT_MAX_ATTEMPTS = 3
# A correction request shows the model this many characters before a JSON error.
JSON_ERROR_CONTEXT_CHARACTERS = 60
# A reply with one of these errors has content, but the content is not the OCR result.
UNUSABLE_REPLY_ERRORS = (RuntimeError, TypeError, LayoutFormatError)
BYTES_PER_MEBIBYTE = 1024 * 1024
# Some providers reject an image when its base64 text is larger than 5 MiB.
MAX_IMAGE_BASE64_BYTES = 5 * BYTES_PER_MEBIBYTE
# Base64 writes 4 characters for each 3 bytes.
MAX_IMAGE_BYTES = MAX_IMAGE_BASE64_BYTES // 4 * 3
# The image types that the backend accepts for a page.
PAGE_IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
# The image type that the backend sends.
IMAGE_MIME_TYPE = "image/jpeg"
JPEG_QUALITY = 90
# Full color resolution keeps the edges of colored text sharp.
JPEG_SUBSAMPLING = "4:4:4"
JPEG_MODES = frozenset({"L", "RGB"})
# The EXIF tag that tells a viewer how to turn the image, and its values that turn it.
EXIF_ORIENTATION_TAG = 0x0112
EXIF_TURNED_ORIENTATIONS = range(2, 9)
# The Pillow modes with integer values start with this name. Of the accepted image types,
# only a gray PNG with 16 bits for each pixel opens in such a mode.
INTEGER_MODE = "I"
SIXTEEN_BIT_GRAY_STEPS_PER_8_BIT_STEP = 256
BILEVEL_MODE = "1"
# The decoded page image and its copies can use this much memory. An RGB page of about
# 44 million pixels has this size.
MAX_IMAGE_WORKING_BYTES = 512 * BYTES_PER_MEBIBYTE
# Pillow stores one pixel of these modes in 1 byte,
ONE_BYTE_MODES = frozenset({"1", "L", "P"})
# one pixel of the modes with this name at the start in 2 bytes, and one pixel of the
# other modes in 4 bytes.
TWO_BYTE_MODE = "I;16"
# A resize to a smaller image makes one temporary image and the result. The two are
# smaller than the image.
RESIZE_COPIES = 2
ALPHA_BAND = "A"
# Pillow puts the transparent palette entry or the transparent color of a file in this field.
TRANSPARENCY_INFO = "transparency"
# A too-large image becomes this much smaller than the size limit needs. The size of a
# JPEG is not exactly proportional to its number of pixels.
IMAGE_SCALE_MARGIN = 0.9
NORMALIZED_COORDINATE_MAX = 1000.0
PIXEL_COORDINATE_TOLERANCE_RATIO = 0.05
# Points a small distance outside the normalized grid also fit a large image. Thus the points
# are read as pixels only when a point of a region is more than this distance outside the grid,
PIXEL_FALLBACK_MIN_DISTANCE = 0.25 * NORMALIZED_COORDINATE_MAX
# and that is true for more than this share of the regions,
PIXEL_FALLBACK_MIN_REGION_SHARE = 0.5
# and for this number of regions or more. One region is not sufficient evidence.
PIXEL_FALLBACK_MIN_REGIONS = 2
# The provider controls this text, so an error message keeps only a bounded part of it.
PROVIDER_DETAIL_MAX_CHARACTERS = 500
TRUNCATION_MARK = "..."
# The values of finish_reason that the chat-completions API defines for an incomplete reply.
FINISH_REASONS = frozenset({"length", "content_filter", "tool_calls", "function_call", "error"})
# Other metadata fields can repeat the request input, so an error message does not have them.
PROVIDER_ERROR_METADATA_FIELDS = ("error_type", "provider_name", "provider_code", "raw")
# The fields of `usage` that have the number of tokens of the request and of the reply.
USAGE_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens")
# The request for the box of an illustration has not more than this many characters of
# its caption. The start of a caption is sufficient to find the illustration.
MAX_CAPTION_HINT_CHARACTERS = 300
REQUEST_OPTIONS = frozenset({"model"})
HTTP_REFERER = "https://github.com/jaysonsantos/scanwich"
APP_TITLE = "Scanwich"
# OpenRouter groups the requests that have the same value of this header.
SESSION_ID_HEADER = "X-Session-Id"

logger = logging.getLogger(__name__)

Result = TypeVar("Result")


@dataclass(frozen=True)
class RequestImage:
    """A page image in the form that the service receives."""

    width: int
    height: int
    data_url: str
    # The page image is this many times larger than the image in the request.
    page_scale_x: float
    page_scale_y: float


class OpenAICompatibleBackend:
    """Vision OCR through an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, *, languages: Sequence[str], options: Mapping[str, Any]) -> None:
        option_values = dict(options)
        self._languages = tuple(languages)
        self._model = _pop_string(option_values, "model", DEFAULT_MODEL)
        aliases = option_values.pop("model_aliases", {})
        if not isinstance(aliases, Mapping) or not all(
            isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip()
            for k, v in aliases.items()
        ):
            raise TypeError("model_aliases must map non-empty aliases to model names")
        self._model = {**DEFAULT_MODEL_ALIASES, **aliases}.get(self._model, self._model)
        self._base_url = _pop_string(option_values, "base_url", DEFAULT_BASE_URL).rstrip("/")
        self._api_key_env = _pop_string(option_values, "api_key_env", DEFAULT_API_KEY_ENV)
        self._timeout = _pop_positive_number(
            option_values,
            "timeout",
            DEFAULT_TIMEOUT_SECONDS,
        )
        self._max_tokens = _pop_positive_integer(
            option_values,
            "max_tokens",
            DEFAULT_MAX_TOKENS,
        )
        self._reasoning_effort = _pop_optional_string(
            option_values,
            "reasoning_effort",
            DEFAULT_REASONING_EFFORT,
        )
        self._max_attempts = _pop_positive_integer(
            option_values,
            "max_attempts",
            DEFAULT_MAX_ATTEMPTS,
        )
        if option_values:
            unsupported = ", ".join(sorted(option_values))
            raise TypeError(f"unsupported OpenAI-compatible backend option(s): {unsupported}")
        self._client: Any | None = None
        self._async_client: Any | None = None
        # One backend instance recognizes the pages of one document, so it is one session.
        self._session_id = str(uuid.uuid4())
        self._session_cost = 0.0
        self._session_requests_with_cost = 0

    def recognize(self, image_path: Path) -> list[OcrRegion]:
        image = _encode_image(image_path)
        return self._complete(
            self._request(image),
            lambda content, attempt: self._parse_reply(content, image, attempt),
        )

    def recognize_layout(self, image_path: Path, *, upright: bool = False) -> PageLayout:
        """Return the blocks of one page in reading order.

        With `upright`, the caller knows that the page is upright in the image. The model
        then gets no request to find the direction of the page.

        The boxes of the layout use the 0-to-1000 grid of the image that the model sees.
        Thus the page image must have no EXIF orientation.
        """
        image = _encode_image(image_path)
        request = self._request(image, layout=True, upright=upright)
        return self._complete(request, self._parse_layout_reply)

    def locate_figure(
        self, image_path: Path, *, estimate: Box | None = None, caption: str | None = None
    ) -> FigureLocation | None:
        """Return the box of one illustration in an image that shows a part of a page.

        The model gives a box that is not exact when it reads a full page. It gives an
        exact box when it gets only the part of the page that has the illustration, and
        only this one task. `estimate` and `caption` tell the model which illustration to
        find when the image shows more than one.
        """
        image = _encode_image(image_path)
        request = self._request(image)
        request["messages"][1]["content"][0]["text"] = self._figure_prompt(
            width=image.width, height=image.height, estimate=estimate, caption=caption
        )
        return self._complete(request, self._parse_figure_reply)

    def _complete(self, request: dict[str, Any], parse: Callable[[str, int], Result]) -> Result:
        """Send a request, and return the result that `parse` makes from the reply.

        `parse` gets the text of the reply and the number of the attempt. When it raises
        an error for an unusable reply, the model gets a request to correct the reply.
        """
        client = self._get_client()
        attempt = 1
        while True:
            logger.info("Sending page image to OpenAI-compatible model %s", self._model)
            started_at = time.monotonic()
            try:
                completion = client.chat.completions.create(**request)
            except Exception as error:
                if attempt > 1:
                    raise self._correction_request_error(error, attempt) from None
                raise RuntimeError(
                    f"OpenAI-compatible request failed for model {self._model}: {error}"
                ) from error
            logger.info(
                "OpenAI-compatible response received after %.1f seconds",
                time.monotonic() - started_at,
            )
            self._log_usage(completion)
            content = self._completion_content(completion, provider_text=attempt == 1)
            try:
                return parse(content, attempt)
            except UNUSABLE_REPLY_ERRORS as error:
                self._log_unusable_reply(error, attempt)
                if attempt == self._max_attempts:
                    raise
                request = _correction_request(request, content, error)
                attempt += 1

    async def recognize_async(self, image_path: Path) -> list[OcrRegion]:
        image = await asyncio.to_thread(_encode_image, image_path)
        client = self._get_client(asynchronous=True)
        request = self._request(image)
        attempt = 1
        while True:
            try:
                completion = await client.chat.completions.create(**request)
            except Exception as error:
                if attempt > 1:
                    raise self._correction_request_error(error, attempt) from None
                raise RuntimeError(
                    f"OpenAI-compatible request failed for model {self._model}"
                ) from error
            self._log_usage(completion)
            content = self._completion_content(completion, provider_text=attempt == 1)
            try:
                return self._parse_reply(content, image, attempt)
            except UNUSABLE_REPLY_ERRORS as error:
                self._log_unusable_reply(error, attempt)
                if attempt == self._max_attempts:
                    raise
                request = _correction_request(request, content, error)
                attempt += 1

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None

    async def recognize_text_async(self, image_path: Path) -> str:
        image = await asyncio.to_thread(_encode_image, image_path)
        client = self._get_client(asynchronous=True)
        try:
            completion = await client.chat.completions.create(
                **self._request(image, plain_text=True)
            )
        except Exception as error:
            raise RuntimeError(
                f"OpenAI-compatible request failed for model {self._model}"
            ) from error
        self._log_usage(completion)
        return self._completion_content(completion, allow_empty=True)

    def _request(
        self,
        image: RequestImage,
        *,
        plain_text: bool = False,
        layout: bool = False,
        upright: bool = False,
    ) -> dict[str, Any]:
        width, height = image.width, image.height
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise OCR engine. Return only the requested "
                        "result and never translate, summarize, or infer missing text."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt(width=width, height=height)},
                        {
                            "type": "image_url",
                            "image_url": {"url": image.data_url},
                        },
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": self._max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self._reasoning_effort is not None:
            request["extra_body"] = {"reasoning_effort": self._reasoning_effort}
        if plain_text:
            request.pop("response_format")
            request["messages"][1]["content"][0]["text"] = (
                f"Transcribe this {width}x{height} pixel document image as plain text. "
                f"Expected language codes: {', '.join(self._languages) or 'unknown'}. "
                "Preserve reading order, line breaks, paragraphs, spacing, spelling, punctuation, "
                "and diacritics. Return only the document text, without JSON, Markdown fences, "
                "coordinates, or commentary. Return an empty string if no text is visible."
            )
        if layout:
            request["messages"][1]["content"][0]["text"] = self._layout_prompt(
                width=width, height=height, upright=upright
            )

        return request

    def _parse_reply(self, content: str, image: RequestImage, attempt: int) -> list[OcrRegion]:
        payload = _parse_json_payload(content)
        regions = _parse_regions(payload, image_width=image.width, image_height=image.height)
        self._log_corrected_reply(attempt)
        return _scale_regions(regions, image)

    def _parse_layout_reply(self, content: str, attempt: int) -> PageLayout:
        layout = page_layout_from_json(_parse_json_payload(content), reply=True)
        self._log_corrected_reply(attempt)
        return layout

    def _parse_figure_reply(self, content: str, attempt: int) -> FigureLocation | None:
        location = figure_location_from_json(_parse_json_payload(content))
        self._log_corrected_reply(attempt)
        return location

    def _log_corrected_reply(self, attempt: int) -> None:
        if attempt > 1:
            logger.info(
                "OpenAI-compatible model %s corrected its reply in attempt %d of %d",
                self._model,
                attempt,
                self._max_attempts,
            )

    def _log_unusable_reply(self, error: Exception, attempt: int) -> None:
        logger.warning(
            "OpenAI-compatible model %s returned an unusable reply in attempt %d of %d (%s: %s)",
            self._model,
            attempt,
            self._max_attempts,
            type(error).__name__,
            _defect_summary(error),
        )

    def _correction_request_error(self, error: Exception, attempt: int) -> RuntimeError:
        """Return the error for a correction request that failed, without provider text.

        A correction request has text from the model reply, and a provider can repeat request
        input in its error.
        """
        status_code = getattr(error, "status_code", None)
        status = f", HTTP status {status_code}" if isinstance(status_code, int) else ""
        return RuntimeError(
            f"OpenAI-compatible request failed for model {self._model} in attempt {attempt} of "
            f"{self._max_attempts}: {type(error).__name__}{status}"
        )

    def _completion_content(
        self, completion: Any, *, allow_empty: bool = False, provider_text: bool = True
    ) -> str:
        """Return the text of the reply.

        A correction request has text from the model reply, and a provider can repeat request
        input in an error. Thus an error has text from the provider only when
        `provider_text` is true.
        """
        choices = getattr(completion, "choices", None)
        if not choices:
            raise RuntimeError(
                "OpenAI-compatible service returned no completion choices"
                + (
                    _provider_error_detail(getattr(completion, "error", None))
                    if provider_text
                    else ""
                )
            )
        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason not in {None, "stop"}:
            if provider_text:
                detail = (
                    f"finish_reason={finish_reason!r}"
                    + _native_finish_reason_detail(getattr(choice, "native_finish_reason", None))
                    + _provider_error_detail(getattr(choice, "error", None))
                )
            elif finish_reason in FINISH_REASONS:
                detail = f"finish_reason={finish_reason!r}"
            else:
                detail = "finish_reason=unknown"
            raise RuntimeError(
                "OpenAI-compatible completion ended before OCR was complete: " + detail
            )
        content = getattr(getattr(choice, "message", None), "content", None)
        if not isinstance(content, str) or (not allow_empty and not content.strip()):
            raise RuntimeError(
                "OpenAI-compatible service returned an empty or unsupported response"
            )
        return content

    def _log_usage(self, completion: Any) -> None:
        """Log the tokens and the cost that the service reports for one request."""
        usage = getattr(completion, "usage", None)
        tokens = [getattr(usage, name, None) for name in USAGE_TOKEN_FIELDS]
        if all(_is_count(count) for count in tokens):
            # The reasoning tokens are a part of the completion tokens. A model that
            # reasons much is slow and has a high cost for a short reply.
            details = getattr(usage, "completion_tokens_details", None)
            reasoning = (
                details.get("reasoning_tokens")
                if isinstance(details, Mapping)
                else getattr(details, "reasoning_tokens", None)
            )
            logger.info(
                "OpenAI-compatible session %s, model %s: usage.prompt_tokens=%d, "
                "usage.completion_tokens=%d%s for this request",
                self._session_id,
                self._model,
                *tokens,
                f", reasoning_tokens={reasoning}" if _is_count(reasoning) else "",
            )
        self._log_cost(completion)

    def _log_cost(self, completion: Any) -> None:
        """Log the cost that the service reports for one request, and the session total."""
        cost = getattr(getattr(completion, "usage", None), "cost", None)
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            return
        if not math.isfinite(cost) or cost < 0:
            return
        self._session_cost += cost
        self._session_requests_with_cost += 1
        logger.info(
            "OpenAI-compatible session %s, model %s: usage.cost=%.6f for this request, "
            "total %.6f for %d request(s)",
            self._session_id,
            self._model,
            cost,
            self._session_cost,
            self._session_requests_with_cost,
        )

    def _get_client(self, *, asynchronous: bool = False) -> Any:
        cached = self._async_client if asynchronous else self._client
        if cached is not None:
            return cached

        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            raise RuntimeError(
                f"the OpenAI-compatible backend requires {self._api_key_env} to be set"
            )
        logger.info("Initializing OpenAI-compatible client for model %s", self._model)
        try:
            if asynchronous:
                from openai import AsyncOpenAI as Client
            else:
                from openai import OpenAI as Client
        except ImportError as error:
            raise RuntimeError(
                "the OpenAI-compatible backend requires the 'openai' package"
            ) from error

        client_options: dict[str, Any] = {
            "api_key": api_key,
            "base_url": self._base_url,
            "timeout": self._timeout,
        }
        if self._base_url == DEFAULT_BASE_URL:
            client_options["default_headers"] = {
                "HTTP-Referer": HTTP_REFERER,
                "X-Title": APP_TITLE,
                SESSION_ID_HEADER: self._session_id,
            }
        client = Client(**client_options)
        if asynchronous:
            self._async_client = client
        else:
            self._client = client
        return client

    def _prompt(self, *, width: int, height: int) -> str:
        languages = ", ".join(self._languages) or "unknown"
        return (
            f"Transcribe every visible text line in this {width}x{height} pixel document image. "
            f"Expected language codes: {languages}. Preserve spelling, case, punctuation, numbers, "
            "and diacritics exactly. Return regions in natural reading order. For each region, "
            "return a tight four-point polygon ordered top-left, top-right, bottom-right, "
            "bottom-left. Use coordinate_system=normalized and express x and y on a normalized "
            "0-to-1000 grid, where (0, 0) is the "
            "image's top-left and (1000, 1000) is its bottom-right. Return an empty regions "
            "array when no text is visible. Return one JSON object with exactly this shape: "
            '{"coordinate_system":"normalized","regions":[{"text":"visible text",'
            '"polygon":[{"x":0,"y":0},{"x":1000,"y":0},{"x":1000,"y":1000},'
            '{"x":0,"y":1000}]}]}. The coordinate_system may instead be "pixels" only '
            "when every coordinate is expressed in native image pixels."
        )

    def _layout_prompt(self, *, width: int, height: int, upright: bool = False) -> str:
        """Return the request for the layout of a page, in the JSON form of `PageLayout`.

        With `upright`, the caller knows the direction of the page. The request then has
        no step that finds the direction, so the model reads the page in each case.
        """
        languages = ", ".join(self._languages) or "unknown"
        box = '{"left":L,"top":T,"right":R,"bottom":B}'
        if upright:
            shape = f'{{"content_box":{box},"blocks":[...]}}'
            direction = "Direction of the page. The page is upright in the image.\n\n"
            no_content = '{"blocks":[]}'
        else:
            shape = f'{{"page_top":"top","content_box":{box},"blocks":[...]}}'
            direction = (
                'Direction of the page. "page_top" is the side of the image where the top of '
                'the page is: "top", "right", "bottom", or "left". When it is not "top", do '
                'not read the page. Return only {"page_top":"<side>","blocks":[]}. You then '
                "get the image again, turned upright.\n\n"
            )
            no_content = '{"page_top":"top","blocks":[]}'
        return (
            f"This {width}x{height} pixel image shows one page of a printed document. "
            "Convert the page to structured data for an archive copy of the document. "
            f"Expected language codes: {languages}.\n\n"
            f"Return one JSON object with this shape: {shape}\n\n"
            f"{direction}"
            f"Coordinates. A box is {box} on a 0-to-1000 grid, where (0, 0) is the top-left "
            "corner of the image and (1000, 1000) is its bottom-right corner. L and R are "
            "distances from the left edge of the image. T and B are distances from its top "
            'edge. "content_box" is the box of the printed area of the page: all its text '
            "and illustrations, without the margins.\n\n"
            "Scope. Read only the page that fills most of the image. Ignore the background, "
            "the binding, fingers, and the visible parts of other pages. Do not translate, "
            "correct, modernize, or complete the text. Keep the spelling, the capital letters, "
            "the punctuation, the numbers, and the diacritics of the page. Do not add an "
            "accent that the page does not print.\n\n"
            "Capital forms. Write each letter in the case that the page prints. A line in "
            "capitals or in small capitals has capital forms: write its letters as capital "
            "letters, also in a caption and in a running head. Do not change such a line to "
            "lowercase letters or to sentence case. Small capitals are capital forms that "
            "are as high as lowercase letters. Put only them in <sc>, and write a large "
            "capital letter before the tag. For a caption that the page prints as TABLE II. "
            "RESULTS OF THE TESTS, with a large T and a large R, write "
            '"T<sc>ABLE</sc> II. R<sc>ESULTS OF THE TESTS</sc>" and not "Table II. Results '
            'of the tests".\n\n'
            'Blocks. "blocks" has the parts of the page in reading order. Each block is one '
            "of these, with only the fields that are shown:\n"
            '- {"type":"page_header","text":"...","position":"top-center"}, '
            '{"type":"page_footer","text":"...","position":"bottom-center"}, and '
            '{"type":"page_number","text":"...","position":"top-right"}: the running head, '
            "the running foot, and the page number that the page prints outside its text. "
            '"position" is the place on the page: "top-left", "top-center", "top-right", '
            '"bottom-left", "bottom-center", or "bottom-right".\n'
            '- {"type":"heading","level":1,"text":"..."}: level 1 is a chapter or the main '
            'title, level 2 is a section, and level 3 is a subsection. Add "align":"center" '
            "for a centered heading.\n"
            '- {"type":"paragraph","text":"..."}: one paragraph. Write its lines as one line '
            "of text, with a space between the lines. When the end of a line divides a word "
            "with a hyphen, write that word in one piece and without that hyphen. When the "
            "last line of the page ends with such a hyphen, keep that hyphen at the end of "
            "the text: the rest of the word is on the next page. Add "
            '"align":"center" or "align":"right" for a paragraph that the page centers or '
            'sets flush right. Add "continued":true when the first paragraph of the page '
            "continues a paragraph of the page before.\n"
            '- {"type":"quote","text":"..."}: a block quotation that the page indents.\n'
            '- {"type":"list","ordered":false,"items":[{"marker":"a)","text":"..."}]}: a '
            'list. "marker" is the marker that the page prints. A list in a list is not '
            "possible: write its items as items of the one list, in reading order.\n"
            '- {"type":"table","caption":"...","header_rows":1,"columns":["left","right"],'
            '"rules":"horizontal","rows":[["cell",{"text":"cell","colspan":2}]]}: a table. '
            '"rows" has all rows, and the header rows are the first ones. Each row is an '
            'array of cells. A cell is a string, or an object with "text" and optionally '
            '"colspan" and "rowspan". Write an empty cell as "". Keep '
            'ditto marks. "columns" has the alignment of each column: "left", "center", or '
            '"right". "rules" tells which rules the table prints: "none", "horizontal" '
            '(rules above the table, below the header, and below the table), "columns" (also '
            'vertical rules between the columns), or "grid" (a rule around each cell). Add '
            '"double_rule":true when the rule above the table and the rule below it are '
            "double rules.\n"
            f'- {{"type":"figure","box":{box},"caption":"..."}}: an '
            "illustration, photograph, diagram, map, ornament, stamp, or signature. "
            '"box" contains the full picture with its labels, and not its caption. Do not '
            "write the text inside a picture.\n"
            '- {"type":"footnote","mark":"1","text":"..."}: a footnote, with its reference '
            "mark as the page prints it.\n"
            'Leave out "caption" when there is none.\n\n'
            'Inline markup. In "text", in "caption", and in table cells, use only these tags: '
            "<b> bold, <i> italic, <u> underlined, <sc> small capitals, <sup> superscript, "
            "<sub> subscript, <br> for a line break that the page makes on purpose (verse, "
            "an address, the lines of a title, the lines of a centered caption), and "
            "<fn>mark</fn> at the place of a footnote reference mark. The text in <sc> has "
            "capital letters, as the paragraph about capital forms tells. <fn> has "
            "only the mark. The text of the footnote is in its footnote block. Do not use "
            "Markdown, LaTeX, or other HTML. Write a formula as plain text with Unicode "
            "characters.\n\n"
            f"Return {no_content} for a page that has no content. Return only the JSON object."
        )

    def _figure_prompt(
        self, *, width: int, height: int, estimate: Box | None, caption: str | None
    ) -> str:
        """Return the request for the box of one illustration in a part of a page."""
        box = '{"left":L,"top":T,"right":R,"bottom":B}'
        hints = []
        caption = " ".join(plain_text(caption or "").split())[:MAX_CAPTION_HINT_CHARACTERS]
        if caption:
            quoted = json.dumps(caption, ensure_ascii=False)
            hints.append(f"The page gives this caption to the illustration: {quoted}.")
        if estimate is not None:
            # The model gets only the center of the estimate. The sides of the estimate
            # are the values that this request must correct.
            hints.append(
                "A first estimate puts the center of the illustration near "
                f"x={(estimate.left + estimate.right) / 2:.0f}, "
                f"y={(estimate.top + estimate.bottom) / 2:.0f} on the grid. The estimate is "
                "not exact."
            )
        which = ""
        if hints:
            hints.append(
                "When the image shows more than one illustration, find only the one that "
                "these data describe."
            )
            which = "Which illustration. " + " ".join(hints) + "\n\n"
        return (
            f"This {width}x{height} pixel image shows a part of one page of a printed "
            "document. The part has an illustration: a drawing, photograph, diagram, map, "
            "ornament, stamp, or signature. It can also have text of the page and parts of "
            "other illustrations around it.\n\n"
            f'Return one JSON object with this shape: {{"box":{box},"cut":[]}}\n\n'
            f"Coordinates. A box is {box} on a 0-to-1000 grid, where (0, 0) is the top-left "
            "corner of the image and (1000, 1000) is its bottom-right corner. L and R are "
            "distances from the left edge of the image. T and B are distances from its top "
            "edge.\n\n"
            f"{which}"
            '"box" contains the full illustration: all its lines, and all the labels and '
            "numbers that are a part of it. It does not contain the caption that the page "
            "prints above or below the illustration, and it does not contain the paragraphs "
            "and tables of the page. Find the outermost mark of the illustration at each of "
            "its four sides. When you are not sure that a mark is a part of the "
            "illustration, include it.\n\n"
            '"cut" has the sides of the image where the illustration continues outside the '
            'image: "left", "top", "right", "bottom". It is empty when the image shows the '
            "full illustration.\n\n"
            'Return {"box":null,"cut":[]} when the image shows no illustration. Return only '
            "the JSON object."
        )


def _encode_image(image_path: Path) -> RequestImage:
    """Return the page image as a JPEG that is not larger than the provider limit."""
    try:
        with Image.open(image_path) as page_image:
            page_width, page_height = page_image.size
            if page_width <= 0 or page_height <= 0:
                raise RuntimeError(
                    f"OCR page image has invalid dimensions: {page_width}x{page_height}"
                )
            mime_type = Image.MIME.get(page_image.format or "")
            if mime_type not in PAGE_IMAGE_MIME_TYPES:
                raise RuntimeError(
                    "OpenAI-compatible service does not support image type "
                    f"{mime_type or 'unknown'}"
                )
            # Pillow knows the size and the mode before it decodes the pixels.
            working_bytes = _working_bytes(page_image)
            if working_bytes > MAX_IMAGE_WORKING_BYTES:
                raise RuntimeError(
                    f"OCR page image of {page_width}x{page_height} pixels needs "
                    f"{math.ceil(working_bytes / BYTES_PER_MEBIBYTE)} MiB of memory, and the limit "
                    f"is {MAX_IMAGE_WORKING_BYTES // BYTES_PER_MEBIBYTE} MiB; "
                    "use a lower DPI or a smaller image"
                )
            orientation = _exif_orientation(page_image)
            image = _jpeg_mode_image(page_image)
            jpeg = _jpeg_bytes(image, orientation)
            while len(jpeg) > MAX_IMAGE_BYTES and image.size != (1, 1):
                ratio = math.sqrt(MAX_IMAGE_BYTES / len(jpeg)) * IMAGE_SCALE_MARGIN
                size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
                image = image.resize(size, Image.Resampling.LANCZOS)
                jpeg = _jpeg_bytes(image, orientation)
    except OSError as error:
        raise RuntimeError(f"could not read OCR page image {image_path}: {error}") from error
    if image.size != (page_width, page_height):
        logger.warning(
            "OpenAI-compatible page image of %dx%d pixels is too large for one request; "
            "sending it as %dx%d pixels (%d bytes)",
            page_width,
            page_height,
            image.width,
            image.height,
            len(jpeg),
        )
    encoded = base64.b64encode(jpeg).decode("ascii")
    return RequestImage(
        width=image.width,
        height=image.height,
        data_url=f"data:{IMAGE_MIME_TYPE};base64,{encoded}",
        page_scale_x=page_width / image.width,
        page_scale_y=page_height / image.height,
    )


def _jpeg_mode_image(image: Image.Image) -> Image.Image:
    """Return the image in a mode that JPEG has. White replaces transparent pixels.

    `_conversion_copy_modes` gives the copies that this function makes.
    """
    if image.mode in JPEG_MODES:
        return image
    if image.mode.startswith(INTEGER_MODE):
        # Pillow does not scale these values in a conversion. It limits them to 255.
        to_8_bits = 1 / SIXTEEN_BIT_GRAY_STEPS_PER_8_BIT_STEP
        return image.point(lambda value: value * to_8_bits).convert("L")
    if image.mode == BILEVEL_MODE:
        return image.convert("L")
    if not _has_transparency(image):
        return image.convert("RGB")
    # A conversion to the mode of the image makes a copy that is not necessary.
    rgba = image if image.mode == "RGBA" else image.convert("RGBA")
    opaque = Image.new("RGB", rgba.size, "white")
    opaque.paste(rgba, mask=rgba)
    return opaque


def _conversion_copy_modes(image: Image.Image) -> tuple[str, ...]:
    """Return the modes of the copies that `_jpeg_mode_image` makes. The last one is the result."""
    if image.mode in JPEG_MODES:
        return ()
    if image.mode.startswith(INTEGER_MODE):
        return (image.mode, "L")
    if image.mode == BILEVEL_MODE:
        return ("L",)
    if not _has_transparency(image) or image.mode == "RGBA":
        return ("RGB",)
    return ("RGBA", "RGB")


def _has_transparency(image: Image.Image) -> bool:
    return ALPHA_BAND in image.getbands() or TRANSPARENCY_INFO in image.info


def _pixel_bytes(mode: str) -> int:
    """Return the number of bytes that Pillow uses for one pixel of the mode."""
    if mode in ONE_BYTE_MODES:
        return 1
    if mode.startswith(TWO_BYTE_MODE):
        return 2
    return 4


def _working_bytes(image: Image.Image) -> int:
    """Return an upper limit of the memory for the pixels of the page image and its copies.

    The copies are the copies of the conversion to a JPEG mode, and the copies of a resize
    when the JPEG is too large.
    """
    copy_modes = _conversion_copy_modes(image)
    jpeg_mode = copy_modes[-1] if copy_modes else image.mode
    bytes_per_pixel = (
        _pixel_bytes(image.mode)
        + sum(_pixel_bytes(mode) for mode in copy_modes)
        + RESIZE_COPIES * _pixel_bytes(jpeg_mode)
    )
    return image.width * image.height * bytes_per_pixel


def _exif_orientation(image: Image.Image) -> int | None:
    """Return the EXIF orientation of the image when it tells a viewer to turn the image."""
    try:
        orientation = image.getexif().get(EXIF_ORIENTATION_TAG)
    except Exception:
        # Damaged metadata must not stop the OCR of the page. The record has no error text,
        # because that text can have metadata of the image.
        logger.debug("Could not read the EXIF data of a page image")
        return None
    return orientation if orientation in EXIF_TURNED_ORIENTATIONS else None


def _jpeg_bytes(image: Image.Image, orientation: int | None) -> bytes:
    """Return the image as a JPEG. The JPEG has the EXIF orientation and no other metadata."""
    options: dict[str, Any] = {}
    if orientation is not None:
        # The service got this tag with the page image file before the backend made a JPEG.
        # Without the tag, the service gets a turned page.
        exif = Image.Exif()
        exif[EXIF_ORIENTATION_TAG] = orientation
        options["exif"] = exif
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING, **options)
    return output.getvalue()


def _scale_regions(regions: list[OcrRegion], image: RequestImage) -> list[OcrRegion]:
    """Convert regions from the pixels of the request image to the pixels of the page image."""
    if image.page_scale_x == 1 and image.page_scale_y == 1:
        return regions
    return [
        replace(
            region,
            polygon=tuple(
                Point(x=point.x * image.page_scale_x, y=point.y * image.page_scale_y)
                for point in region.polygon
            ),
        )
        for region in regions
    ]


def _correction_request(request: dict[str, Any], content: str, error: Exception) -> dict[str, Any]:
    """Return the request that shows the model its unusable reply and asks for a new one.

    With a temperature of 0, the model mostly repeats the defect for the same request.
    Thus the model gets its reply and the defect.
    """
    return {
        **request,
        "messages": [
            *request["messages"],
            {"role": "assistant", "content": content},
            {"role": "user", "content": _correction_prompt(error)},
        ],
    }


def _defect_summary(error: Exception) -> str:
    """Describe an unusable reply for a log record, without text of the reply."""
    if isinstance(error.__cause__, json.JSONDecodeError):
        return f"{error}: {error.__cause__}"
    return str(error)


def _correction_prompt(error: Exception) -> str:
    """Describe an unusable reply to the model. This text can have parts of the reply."""
    cause = error.__cause__
    if isinstance(cause, json.JSONDecodeError):
        context = cause.doc[max(0, cause.pos - JSON_ERROR_CONTEXT_CHARACTERS) : cause.pos]
        defect = (
            f"it is not valid JSON ({cause.msg}). "
            f"The JSON parser stopped directly after this text: {context}"
        )
    else:
        defect = str(error)
        # The hint of the layout check names the fields of the reply that are wrong. Only
        # the model gets it: a log record has the message of the error and not the hint.
        hint = getattr(error, "hint", "")
        if hint:
            defect = f"{defect}. {hint}"
    return (
        f"Your last reply cannot be used: {defect}\n"
        "Return the complete JSON object again, in the shape that the first message gives. "
        'In a JSON string, write a double quote as \\" and a backslash as \\\\. '
        "Return only the JSON object."
    )


def _provider_error_detail(error: Any) -> str:
    """Describe a provider error object by its code, its message, and named metadata fields."""
    if not isinstance(error, Mapping):
        return ""
    details: list[str] = []
    code = error.get("code")
    if isinstance(code, (int, str)) and not isinstance(code, bool):
        details.append(f"code={code!r}")
    message = error.get("message")
    if isinstance(message, str):
        details.append(f"message={message!r}")
    metadata = error.get("metadata")
    if isinstance(metadata, Mapping):
        details.extend(
            f"{name}={metadata[name]!r}"
            for name in PROVIDER_ERROR_METADATA_FIELDS
            if metadata.get(name) is not None
        )
    if not details:
        return ""
    return _limit_provider_detail(": provider error " + ", ".join(details))


def _native_finish_reason_detail(native_finish_reason: Any) -> str:
    if not isinstance(native_finish_reason, str):
        return ""
    return _limit_provider_detail(f", native_finish_reason={native_finish_reason!r}")


def _limit_provider_detail(detail: str) -> str:
    if len(detail) <= PROVIDER_DETAIL_MAX_CHARACTERS:
        return detail
    return detail[: PROVIDER_DETAIL_MAX_CHARACTERS - len(TRUNCATION_MARK)] + TRUNCATION_MARK


def _parse_json_payload(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.debug(
            "OpenAI-compatible returned non-JSON content; attempting fallback parse (%d chars)",
            len(content),
        )

    stripped = content.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()

    object_start = stripped.find("{")
    if object_start == -1:
        raise RuntimeError("OpenAI-compatible service returned invalid JSON")
    try:
        payload, end = json.JSONDecoder().raw_decode(stripped[object_start:])
    except json.JSONDecodeError as error:
        raise RuntimeError("OpenAI-compatible service returned invalid JSON") from error
    remainder = stripped[object_start + end :]
    if remainder.strip():
        raise RuntimeError("OpenAI-compatible service returned invalid JSON with trailing text")
    logger.warning("OpenAI-compatible service returned non-JSON output; fallback parser used")
    return payload


def _parse_regions(payload: Any, *, image_width: int, image_height: int) -> list[OcrRegion]:
    if not isinstance(payload, Mapping):
        raise TypeError("OpenAI-compatible OCR response must be a JSON object")
    coordinate_system = payload.get("coordinate_system")
    if not isinstance(coordinate_system, str) or coordinate_system not in {"normalized", "pixels"}:
        raise TypeError(
            "OpenAI-compatible OCR response must declare coordinate_system as "
            "'normalized' or 'pixels'"
        )
    raw_regions = payload.get("regions")
    if not isinstance(raw_regions, list):
        raise TypeError("OpenAI-compatible OCR response must contain a regions array")

    raw_parsed_regions: list[tuple[str, tuple[Point, ...]]] = []
    for region_index, raw_region in enumerate(raw_regions, start=1):
        if not isinstance(raw_region, Mapping):
            raise TypeError(f"OpenAI-compatible OCR region {region_index} must be an object")
        text = raw_region.get("text")
        if not isinstance(text, str):
            raise TypeError(f"OpenAI-compatible OCR region {region_index} text must be a string")
        if not text.strip():
            raise RuntimeError(f"OpenAI-compatible OCR region {region_index} has no text")
        raw_polygon = raw_region.get("polygon")
        if not isinstance(raw_polygon, list):
            raise TypeError(f"OpenAI-compatible OCR region {region_index} polygon must be an array")
        if len(raw_polygon) != 4:
            raise RuntimeError(
                f"OpenAI-compatible OCR region {region_index} polygon must contain four points"
            )
        points = tuple(
            _parse_raw_point(
                raw_point,
                region_index=region_index,
                point_index=point_index,
            )
            for point_index, raw_point in enumerate(raw_polygon, start=1)
        )
        raw_parsed_regions.append((text.strip(), points))

    used_coordinate_system = _select_coordinate_system(
        raw_parsed_regions,
        declared=coordinate_system,
        image_width=image_width,
        image_height=image_height,
    )
    _log_points_outside_bounds(
        raw_parsed_regions,
        declared=coordinate_system,
        coordinate_system=used_coordinate_system,
        image_width=image_width,
        image_height=image_height,
    )
    regions: list[OcrRegion] = []
    for region_index, (text, raw_points) in enumerate(raw_parsed_regions, start=1):
        points = tuple(
            _convert_point(
                point,
                coordinate_system=used_coordinate_system,
                region_index=region_index,
                point_index=point_index,
                image_width=image_width,
                image_height=image_height,
            )
            for point_index, point in enumerate(raw_points, start=1)
        )
        regions.append(
            OcrRegion(
                text=text,
                polygon=points,  # type: ignore[arg-type]
            )
        )
    return regions


def _parse_raw_point(
    raw_point: Any,
    *,
    region_index: int,
    point_index: int,
) -> Point:
    if not isinstance(raw_point, Mapping):
        raise TypeError(
            f"OpenAI-compatible OCR region {region_index} point {point_index} must be an object"
        )
    x = _parse_coordinate(raw_point.get("x"), region_index=region_index, point_index=point_index)
    y = _parse_coordinate(raw_point.get("y"), region_index=region_index, point_index=point_index)
    return Point(x=x, y=y)


def _parse_coordinate(value: Any, *, region_index: int, point_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"OpenAI-compatible OCR region {region_index} point {point_index} has a non-numeric coordinate"
        )
    coordinate = float(value)
    if not math.isfinite(coordinate):
        raise RuntimeError(
            f"OpenAI-compatible OCR region {region_index} point {point_index} is not finite"
        )
    return coordinate


def _select_coordinate_system(
    regions: Sequence[tuple[str, tuple[Point, ...]]],
    *,
    declared: str,
    image_width: int,
    image_height: int,
) -> str:
    """Return the coordinate system of the points, which can differ from the declared one."""
    if declared == "pixels":
        logger.warning(
            "OpenAI-compatible service returned native pixel coordinates; using them directly"
        )
        return declared

    polygons = [polygon for _, polygon in regions]
    regions_outside_grid = sum(
        not _points_fit(
            polygon,
            maximum_x=NORMALIZED_COORDINATE_MAX,
            maximum_y=NORMALIZED_COORDINATE_MAX,
            tolerance_x=PIXEL_FALLBACK_MIN_DISTANCE,
            tolerance_y=PIXEL_FALLBACK_MIN_DISTANCE,
        )
        for polygon in polygons
    )
    if (
        regions_outside_grid < PIXEL_FALLBACK_MIN_REGIONS
        or regions_outside_grid <= len(polygons) * PIXEL_FALLBACK_MIN_REGION_SHARE
    ):
        return declared
    if not _points_fit(
        [point for polygon in polygons for point in polygon],
        maximum_x=float(image_width),
        maximum_y=float(image_height),
        tolerance_x=image_width * PIXEL_COORDINATE_TOLERANCE_RATIO,
        tolerance_y=image_height * PIXEL_COORDINATE_TOLERANCE_RATIO,
    ):
        return declared
    logger.warning(
        "OpenAI-compatible service declared normalized coordinates, but %d of %d regions are "
        "more than %d units outside the 0-to-%d grid and all points fit the %dx%d pixel image; "
        "using the coordinates as pixels",
        regions_outside_grid,
        len(polygons),
        PIXEL_FALLBACK_MIN_DISTANCE,
        NORMALIZED_COORDINATE_MAX,
        image_width,
        image_height,
    )
    return "pixels"


def _log_points_outside_bounds(
    regions: Sequence[tuple[str, tuple[Point, ...]]],
    *,
    declared: str,
    coordinate_system: str,
    image_width: int,
    image_height: int,
) -> None:
    """Write one warning for a page that has points outside the bounds of its coordinates."""
    maximum_x, maximum_y = _coordinate_bounds(
        coordinate_system, image_width=image_width, image_height=image_height
    )
    points = [point for _, polygon in regions for point in polygon]
    distances = [
        distance
        for point in points
        if (distance := max(-point.x, point.x - maximum_x, -point.y, point.y - maximum_y)) > 0
    ]
    if distances:
        logger.warning(
            "OpenAI-compatible OCR coordinates are outside their bounds and were clamped: "
            "declared coordinate_system=%s, used coordinate_system=%s, image=%dx%d pixels, "
            "%d of %d points outside, largest distance %.1f",
            declared,
            coordinate_system,
            image_width,
            image_height,
            len(distances),
            len(points),
            max(distances),
        )


def _coordinate_bounds(
    coordinate_system: str, *, image_width: int, image_height: int
) -> tuple[float, float]:
    if coordinate_system == "normalized":
        return NORMALIZED_COORDINATE_MAX, NORMALIZED_COORDINATE_MAX
    return float(image_width), float(image_height)


def _points_fit(
    points: Sequence[Point],
    *,
    maximum_x: float,
    maximum_y: float,
    tolerance_x: float,
    tolerance_y: float,
) -> bool:
    return all(
        -tolerance_x <= point.x <= maximum_x + tolerance_x
        and -tolerance_y <= point.y <= maximum_y + tolerance_y
        for point in points
    )


def _convert_point(
    point: Point,
    *,
    coordinate_system: str,
    region_index: int,
    point_index: int,
    image_width: int,
    image_height: int,
) -> Point:
    maximum_x, maximum_y = _coordinate_bounds(
        coordinate_system, image_width=image_width, image_height=image_height
    )
    x = _clamp_coordinate(
        point.x,
        maximum=maximum_x,
        region_index=region_index,
        point_index=point_index,
        axis="x",
    )
    y = _clamp_coordinate(
        point.y,
        maximum=maximum_y,
        region_index=region_index,
        point_index=point_index,
        axis="y",
    )
    return Point(x=x * (image_width / maximum_x), y=y * (image_height / maximum_y))


def _clamp_coordinate(
    coordinate: float,
    *,
    maximum: float,
    region_index: int,
    point_index: int,
    axis: str,
) -> float:
    clamped = min(max(coordinate, 0.0), maximum)
    if clamped != coordinate:
        logger.debug(
            "Clamping OpenAI-compatible OCR region %d point %d %s coordinate %.1f to %.1f",
            region_index,
            point_index,
            axis,
            coordinate,
            clamped,
        )
    return clamped


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _pop_string(options: dict[str, Any], name: str, default: str) -> str:
    value = options.pop(name, default)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"OpenAI-compatible's {name!r} backend option must be a non-empty string")
    return value.strip()


def _pop_optional_string(options: dict[str, Any], name: str, default: str) -> str | None:
    value = options.pop(name, default)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"OpenAI-compatible's {name!r} backend option must be a string or null")
    return value.strip()


def _pop_positive_number(options: dict[str, Any], name: str, default: float) -> float:
    value = options.pop(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"OpenAI-compatible's {name!r} backend option must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"OpenAI-compatible's {name!r} backend option must be greater than zero")
    return result


def _pop_positive_integer(options: dict[str, Any], name: str, default: int) -> int:
    value = options.pop(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"OpenAI-compatible's {name!r} backend option must be a positive integer")
    if value <= 0:
        raise ValueError(f"OpenAI-compatible's {name!r} backend option must be greater than zero")
    return value


def factory(
    *,
    languages: Sequence[str],
    options: Mapping[str, Any],
) -> OpenAICompatibleBackend:
    return OpenAICompatibleBackend(languages=languages, options=options)


# HTTP API clients may override the model for one request.
factory.request_options = REQUEST_OPTIONS
