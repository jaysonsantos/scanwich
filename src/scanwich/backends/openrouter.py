from __future__ import annotations

import base64
import json
import logging
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from scanwich.models import OcrRegion, Point

DEFAULT_MODEL = "deepseek/deepseek-v4-flash-vision-exp"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_TOKENS = 65_536
DEFAULT_REASONING_EFFORT = "low"
NORMALIZED_COORDINATE_MAX = 1000.0
NORMALIZED_COORDINATE_TOLERANCE = 50.0
PIXEL_COORDINATE_TOLERANCE_RATIO = 0.05
HTTP_REFERER = "https://github.com/jaysonsantos/scanwich"
APP_TITLE = "Scanwich"

logger = logging.getLogger(__name__)


class OpenRouterBackend:
    """Vision OCR through OpenRouter's OpenAI-compatible chat endpoint."""

    def __init__(self, *, languages: Sequence[str], options: Mapping[str, Any]) -> None:
        option_values = dict(options)
        self._languages = tuple(languages)
        self._model = _pop_string(option_values, "model", DEFAULT_MODEL)
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
        if option_values:
            unsupported = ", ".join(sorted(option_values))
            raise TypeError(f"unsupported OpenRouter backend option(s): {unsupported}")
        self._client: Any | None = None

    def recognize(self, image_path: Path) -> list[OcrRegion]:
        width, height, image_data_url = _encode_image(image_path)
        client = self._get_client()
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise OCR engine. Return only the requested structured "
                        "result and never translate, summarize, or infer missing text."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt(width=width, height=height)},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": self._max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self._reasoning_effort is not None:
            request["reasoning_effort"] = self._reasoning_effort

        logger.info("Sending page image to OpenRouter model %s", self._model)
        started_at = time.monotonic()
        try:
            completion = client.chat.completions.create(**request)
        except Exception as error:
            raise RuntimeError(
                f"OpenRouter request failed for model {self._model}: {error}"
            ) from error
        logger.info(
            "OpenRouter response received after %.1f seconds", time.monotonic() - started_at
        )

        choices = getattr(completion, "choices", None)
        if not choices:
            raise RuntimeError("OpenRouter returned no completion choices")
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenRouter returned an empty or unsupported response")
        payload = _parse_json_payload(content)
        return _parse_regions(payload, image_width=width, image_height=height)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            raise RuntimeError(f"the OpenRouter backend requires {self._api_key_env} to be set")
        logger.info("Initializing OpenRouter client for model %s", self._model)
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("the OpenRouter backend requires the 'openai' package") from error

        self._client = OpenAI(
            api_key=api_key,
            base_url=self._base_url,
            timeout=self._timeout,
            default_headers={
                "HTTP-Referer": HTTP_REFERER,
                "X-OpenRouter-Title": APP_TITLE,
            },
        )
        return self._client

    def _prompt(self, *, width: int, height: int) -> str:
        languages = ", ".join(self._languages) or "unknown"
        return (
            f"Transcribe every visible text line in this {width}x{height} pixel document image. "
            f"Expected language codes: {languages}. Preserve spelling, case, punctuation, numbers, "
            "and diacritics exactly. Return regions in natural reading order. For each region, "
            "return a tight four-point polygon ordered top-left, top-right, bottom-right, "
            "bottom-left. Express x and y on a normalized 0-to-1000 grid, where (0, 0) is the "
            "image's top-left and (1000, 1000) is its bottom-right. Return an empty regions array "
            "when no text is visible. Return one JSON object with exactly this shape: "
            '{"regions":[{"text":"visible text","polygon":[{"x":0,"y":0},'
            '{"x":1000,"y":0},{"x":1000,"y":1000},{"x":0,"y":1000}]}]}.'
        )


def _encode_image(image_path: Path) -> tuple[int, int, str]:
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            mime_type = Image.MIME.get(image.format or "")
        image_bytes = image_path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"could not read OCR page image {image_path}: {error}") from error
    if width <= 0 or height <= 0:
        raise RuntimeError(f"OCR page image has invalid dimensions: {width}x{height}")
    if mime_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise RuntimeError(f"OpenRouter does not support image type {mime_type or 'unknown'}")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return width, height, f"data:{mime_type};base64,{encoded}"


def _parse_json_payload(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    stripped = content.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()

    object_start = stripped.find("{")
    if object_start == -1:
        raise RuntimeError("OpenRouter returned invalid JSON")
    try:
        payload, _ = json.JSONDecoder().raw_decode(stripped[object_start:])
    except json.JSONDecodeError as error:
        raise RuntimeError("OpenRouter returned invalid JSON") from error
    return payload


def _parse_regions(payload: Any, *, image_width: int, image_height: int) -> list[OcrRegion]:
    if not isinstance(payload, Mapping):
        raise TypeError("OpenRouter OCR response must be a JSON object")
    raw_regions = payload.get("regions")
    if not isinstance(raw_regions, list):
        raise TypeError("OpenRouter OCR response must contain a regions array")

    raw_parsed_regions: list[tuple[str, tuple[Point, ...]]] = []
    for region_index, raw_region in enumerate(raw_regions, start=1):
        if not isinstance(raw_region, Mapping):
            raise TypeError(f"OpenRouter OCR region {region_index} must be an object")
        text = raw_region.get("text")
        if not isinstance(text, str):
            raise TypeError(f"OpenRouter OCR region {region_index} text must be a string")
        if not text.strip():
            raise RuntimeError(f"OpenRouter OCR region {region_index} has no text")
        raw_polygon = raw_region.get("polygon")
        if not isinstance(raw_polygon, list):
            raise TypeError(f"OpenRouter OCR region {region_index} polygon must be an array")
        if len(raw_polygon) != 4:
            raise RuntimeError(
                f"OpenRouter OCR region {region_index} polygon must contain four points"
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

    coordinate_system = _select_coordinate_system(
        raw_parsed_regions,
        image_width=image_width,
        image_height=image_height,
    )
    regions: list[OcrRegion] = []
    for region_index, (text, raw_points) in enumerate(raw_parsed_regions, start=1):
        points = tuple(
            _convert_point(
                point,
                coordinate_system=coordinate_system,
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
            f"OpenRouter OCR region {region_index} point {point_index} must be an object"
        )
    x = _parse_coordinate(raw_point.get("x"), region_index=region_index, point_index=point_index)
    y = _parse_coordinate(raw_point.get("y"), region_index=region_index, point_index=point_index)
    return Point(x=x, y=y)


def _parse_coordinate(value: Any, *, region_index: int, point_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"OpenRouter OCR region {region_index} point {point_index} has a non-numeric coordinate"
        )
    coordinate = float(value)
    if not math.isfinite(coordinate):
        raise RuntimeError(
            f"OpenRouter OCR region {region_index} point {point_index} is not finite"
        )
    return coordinate


def _select_coordinate_system(
    regions: Sequence[tuple[str, tuple[Point, ...]]],
    *,
    image_width: int,
    image_height: int,
) -> str:
    points = [point for _, polygon in regions for point in polygon]
    if _points_fit(
        points,
        maximum_x=NORMALIZED_COORDINATE_MAX,
        maximum_y=NORMALIZED_COORDINATE_MAX,
        tolerance_x=NORMALIZED_COORDINATE_TOLERANCE,
        tolerance_y=NORMALIZED_COORDINATE_TOLERANCE,
    ):
        return "normalized"

    if _points_fit(
        points,
        maximum_x=float(image_width),
        maximum_y=float(image_height),
        tolerance_x=image_width * PIXEL_COORDINATE_TOLERANCE_RATIO,
        tolerance_y=image_height * PIXEL_COORDINATE_TOLERANCE_RATIO,
    ):
        logger.warning("OpenRouter returned native pixel coordinates; using them directly")
        return "pixels"

    raise RuntimeError("OpenRouter OCR coordinates exceed both normalized and image bounds")


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
    if coordinate_system == "normalized":
        maximum_x = NORMALIZED_COORDINATE_MAX
        maximum_y = NORMALIZED_COORDINATE_MAX
        scale_x = image_width / NORMALIZED_COORDINATE_MAX
        scale_y = image_height / NORMALIZED_COORDINATE_MAX
    else:
        maximum_x = float(image_width)
        maximum_y = float(image_height)
        scale_x = 1.0
        scale_y = 1.0

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
    return Point(x=x * scale_x, y=y * scale_y)


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
        logger.warning(
            "Clamping OpenRouter OCR region %d point %d %s coordinate %.1f to %.1f",
            region_index,
            point_index,
            axis,
            coordinate,
            clamped,
        )
    return clamped


def _pop_string(options: dict[str, Any], name: str, default: str) -> str:
    value = options.pop(name, default)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"OpenRouter's {name!r} backend option must be a non-empty string")
    return value.strip()


def _pop_optional_string(options: dict[str, Any], name: str, default: str) -> str | None:
    value = options.pop(name, default)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"OpenRouter's {name!r} backend option must be a string or null")
    return value.strip()


def _pop_positive_number(options: dict[str, Any], name: str, default: float) -> float:
    value = options.pop(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"OpenRouter's {name!r} backend option must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"OpenRouter's {name!r} backend option must be greater than zero")
    return result


def _pop_positive_integer(options: dict[str, Any], name: str, default: int) -> int:
    value = options.pop(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"OpenRouter's {name!r} backend option must be a positive integer")
    if value <= 0:
        raise ValueError(f"OpenRouter's {name!r} backend option must be greater than zero")
    return value


def factory(
    *,
    languages: Sequence[str],
    options: Mapping[str, Any],
) -> OpenRouterBackend:
    return OpenRouterBackend(languages=languages, options=options)
