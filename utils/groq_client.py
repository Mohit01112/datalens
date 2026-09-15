
"""
groq_client.py — Thin wrapper around the native Groq API.

DataLens responsibilities:
  - Ask Groq for a structured analysis plan.
  - Ask Groq for reproducible EDA code.
  - Keep numerical calculations and chart rendering inside DataLens.
  - Retry transient API failures.
  - Handle malformed JSON responses safely.
  - Trim large dataset profiles before sending them to Groq.

Groq is used for:
  - Analysis planning
  - Business insights
  - EDA code generation

Pandas/Plotly remain the source of truth for actual calculations and charts.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import time
from typing import Any

from groq import Groq

from utils.prompts import (
    ANALYSIS_PLAN_SYSTEM,
    EDA_CODE_SYSTEM,
    build_analysis_plan_user_message,
    build_eda_code_user_message,
)

logger = logging.getLogger(__name__)

# HTTP errors that are safe to retry.
_RETRYABLE_STATUS_CODES = {
    408,
    409,
    429,
    500,
    502,
    503,
    504,
}

# Maximum profile size sent to Groq.
_MAX_PROFILE_CHARS = 16_000


class GroqClient:
    """
    Native Groq client used by DataLens.

    Expected usage:

        GroqClient(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
        )
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        max_retries: int = 3,
    ):
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing.")

        if not model:
            raise ValueError("GROQ_MODEL is missing.")

        self._client = Groq(api_key=api_key)
        self._model = model
        self._max_retries = max(1, int(max_retries))

        logger.info(
            "GroqClient initialised (model=%s, retries=%d)",
            model,
            self._max_retries,
        )

    # ==================================================================
    # RETRY HANDLING
    # ==================================================================

    def _call_with_retry(self, fn, *args, **kwargs):
        """
        Execute a Groq request with exponential-backoff retry handling.
        """

        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):

            try:
                return fn(*args, **kwargs)

            except Exception as exc:

                last_exc = exc

                status = getattr(
                    exc,
                    "status_code",
                    None,
                )

                response = getattr(
                    exc,
                    "response",
                    None,
                )

                if response is not None:
                    status = getattr(
                        response,
                        "status_code",
                        status,
                    )

                if (
                    status in _RETRYABLE_STATUS_CODES
                    and attempt < self._max_retries
                ):

                    wait = 2 ** (attempt - 1)

                    logger.warning(
                        "Retryable Groq error "
                        "(status=%s, attempt=%d/%d). "
                        "Retrying in %ds: %s",
                        status,
                        attempt,
                        self._max_retries,
                        wait,
                        exc,
                    )

                    time.sleep(wait)
                    continue

                logger.exception(
                    "Groq API request failed."
                )

                raise

        raise last_exc or RuntimeError(
            "Groq API request failed."
        )

    # ==================================================================
    # PROFILE TRIMMING
    # ==================================================================

    @staticmethod
    def _trim_profile(
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Trim a large dataset profile before sending it to Groq.
        """

        trimmed = copy.deepcopy(profile)

        original_size = len(
            json.dumps(
                trimmed,
                default=str,
            )
        )

        if original_size <= _MAX_PROFILE_CHARS:
            return trimmed

        # Keep only a few sample rows.
        if "sample_rows" in trimmed:
            trimmed["sample_rows"] = (
                trimmed["sample_rows"][:3]
            )

        # Keep top five categorical values.
        categorical = trimmed.get(
            "categorical_top_values"
        )

        if isinstance(categorical, dict):

            for column, values in categorical.items():

                if isinstance(values, dict):

                    items = list(
                        values.items()
                    )[:5]

                    categorical[column] = dict(
                        items
                    )

        current_size = len(
            json.dumps(
                trimmed,
                default=str,
            )
        )

        # Remove less important information if still too large.
        if current_size > _MAX_PROFILE_CHARS:

            trimmed.pop(
                "top_correlations",
                None,
            )

            trimmed.pop(
                "outlier_pct",
                None,
            )

        current_size = len(
            json.dumps(
                trimmed,
                default=str,
            )
        )

        # Final fallback.
        if current_size > _MAX_PROFILE_CHARS:
            trimmed["sample_rows"] = []

        final_size = len(
            json.dumps(
                trimmed,
                default=str,
            )
        )

        logger.info(
            "Profile trimmed: %d -> %d chars",
            original_size,
            final_size,
        )

        return trimmed

    # ==================================================================
    # RESPONSE CLEANING
    # ==================================================================

    @staticmethod
    def _clean_json_response(
        raw: str,
    ) -> str:
        """
        Clean common formatting problems from Groq JSON responses.
        """

        if not raw:
            return ""

        text = raw.strip()

        # Remove Markdown code fences.
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        return text.strip()

    @staticmethod
    def _extract_json_object(
        raw: str,
    ) -> str | None:
        """
        Extract the first complete JSON object from a response.

        Handles responses such as:

            Here is the JSON:
            {...}

        and:

            ```json
            {...}
            ```
        """

        if not raw:
            return None

        text = raw.strip()

        # First attempt: the complete response itself.
        try:
            parsed = json.loads(text)

            if isinstance(parsed, dict):
                return json.dumps(parsed)

        except Exception:
            pass

        # Remove code fences.
        cleaned = GroqClient._clean_json_response(
            text
        )

        try:
            parsed = json.loads(cleaned)

            if isinstance(parsed, dict):
                return json.dumps(parsed)

        except Exception:
            pass

        # Find the first JSON object using brace matching.
        start = cleaned.find("{")

        if start == -1:
            return None

        depth = 0
        in_string = False
        escaped = False

        for index in range(
            start,
            len(cleaned),
        ):

            char = cleaned[index]

            if escaped:
                escaped = False
                continue

            if char == "\\" and in_string:
                escaped = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1

            elif char == "}":
                depth -= 1

                if depth == 0:

                    candidate = cleaned[
                        start:index + 1
                    ]

                    try:

                        parsed = json.loads(
                            candidate
                        )

                        if isinstance(
                            parsed,
                            dict,
                        ):
                            return json.dumps(
                                parsed
                            )

                    except Exception:
                        return None

        return None

    @staticmethod
    def _validate_plan(
        plan: Any,
    ) -> dict[str, Any]:
        """
        Ensure the analysis plan has the structure
        expected by DataLens.
        """

        if not isinstance(plan, dict):
            raise ValueError(
                "Analysis plan must be a JSON object."
            )

        plan.setdefault(
            "kpis",
            [],
        )

        plan.setdefault(
            "charts",
            [],
        )

        plan.setdefault(
            "insights",
            [],
        )

        if not isinstance(
            plan["kpis"],
            list,
        ):
            plan["kpis"] = []

        if not isinstance(
            plan["charts"],
            list,
        ):
            plan["charts"] = []

        if not isinstance(
            plan["insights"],
            list,
        ):
            plan["insights"] = []

        return plan

    @staticmethod
    def _fallback_plan() -> dict[str, Any]:
        """
        Safe fallback when Groq completely fails to return a usable plan.

        DataLens can still continue without crashing.
        """

        return {
            "kpis": [],
            "charts": [],
            "insights": [],
        }

    # ==================================================================
    # ANALYSIS PLAN
    # ==================================================================

    def get_analysis_plan(
        self,
        profile: dict[str, Any],
        user_request: str,
    ) -> dict[str, Any]:
        """
        Ask Groq what DataLens should analyse.

        Returns:

            {
                "kpis": [...],
                "charts": [...],
                "insights": [...]
            }

        The model is NOT the source of truth for numerical calculations.
        """

        trimmed_profile = self._trim_profile(
            profile
        )

        base_user_message = (
            build_analysis_plan_user_message(
                trimmed_profile,
                user_request,
            )
        )

        guardrails = """
DATA LENS ANALYSIS PLAN RULES

You are generating an analysis PLAN for DataLens.

The dataset profile supplied by the user is the source of truth.

IMPORTANT:
Return ONLY one valid JSON object.

Do not return Markdown.
Do not return ```json.
Do not return Python code.
Do not write any explanation outside the JSON object.

The JSON MUST contain these three top-level keys:

{
  "kpis": [],
  "charts": [],
  "insights": []
}

RULES:

1. Use only columns that exist in the supplied dataset profile.

2. Never invent columns.

3. Never invent numerical values.

4. Never invent categories.

5. Never invent dates.

6. Never invent statistics.

7. KPIs must reference real columns.

8. Charts must reference real columns.

9. DataLens/Pandas calculates the actual numerical KPI and chart values.

10. Do not fabricate KPI values.

11. Insights must be grounded in information available in the dataset profile.

12. If a requested analysis cannot be supported by the available columns,
    omit it.

13. Correlation does not prove causation.

14. Never say that one variable "causes", "drives", "produces",
    or "results in" another variable merely because they are correlated.

15. Prefer wording such as:
    "is positively correlated with"
    "is negatively correlated with"
    "is associated with"
    "tends to be higher when"

16. Do not invent regional or category rankings.

17. Do not describe something as significant unless the supplied data
    supports that conclusion.

18. Keep insights concise and business-focused.

19. Do not generate EDA Python code in the analysis plan.

20. Return valid JSON only.

Example valid response:

{
  "kpis": [
    {
      "name": "Total Revenue",
      "column": "Revenue",
      "agg": "sum"
    }
  ],
  "charts": [
    {
      "type": "bar",
      "x": "Region",
      "y": "Revenue",
      "title": "Revenue by Region"
    }
  ],
  "insights": [
    "Revenue is positively correlated with Units_Sold."
  ]
}
"""

        user_msg = (
            f"{base_user_message}\n\n"
            f"{guardrails}"
        )

        def _call():
            return self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": ANALYSIS_PLAN_SYSTEM,
                    },
                    {
                        "role": "user",
                        "content": user_msg,
                    },
                ],
                temperature=0.0,
            )

        try:

            response = self._call_with_retry(
                _call
            )

            raw = (
                response.choices[0].message.content
                if response.choices
                else ""
            ) or ""

            logger.debug(
                "Raw Groq analysis-plan response:\n%s",
                raw,
            )

            json_text = (
                self._extract_json_object(raw)
            )

            if json_text is not None:

                plan = json.loads(
                    json_text
                )

                return self._validate_plan(
                    plan
                )

            logger.warning(
                "Groq returned malformed analysis-plan JSON. "
                "Attempting a strict JSON retry."
            )

        except Exception as exc:

            logger.warning(
                "Initial analysis-plan request failed: %s",
                exc,
            )

        # ==============================================================
        # STRICT JSON RETRY
        # ==============================================================

        retry_user_msg = f"""
Return ONLY valid JSON.

Do not write any text before or after the JSON.

Do not use Markdown.
Do not use code fences.
Do not return Python.

The required structure is:

{{
  "kpis": [],
  "charts": [],
  "insights": []
}}

User request:
{user_request}

Dataset profile:
{json.dumps(trimmed_profile, default=str)}

Rules:
- Use only columns present in the dataset profile.
- Do not invent values.
- Do not invent columns.
- Do not invent statistics.
- KPIs must reference actual columns.
- Charts must reference actual columns.
- Insights must be based only on the supplied profile.
- Correlation is not causation.
- Do not claim that correlation proves that one variable causes another.
- If information is unavailable, omit that analysis.
- Return ONLY the JSON object.
"""

        def _retry_call():
            return self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict JSON generator. "
                            "Return only valid JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": retry_user_msg,
                    },
                ],
                temperature=0.0,
            )

        try:

            response = self._call_with_retry(
                _retry_call
            )

            raw = (
                response.choices[0].message.content
                if response.choices
                else ""
            ) or ""

            logger.debug(
                "Strict retry response:\n%s",
                raw,
            )

            json_text = (
                self._extract_json_object(raw)
            )

            if json_text is not None:

                plan = json.loads(
                    json_text
                )

                return self._validate_plan(
                    plan
                )

            logger.error(
                "Strict JSON retry also returned malformed JSON."
            )

        except Exception as exc:

            logger.exception(
                "Strict JSON analysis-plan retry failed: %s",
                exc,
            )

        # Never crash Streamlit because the LLM produced malformed JSON.
        logger.warning(
            "Returning safe empty analysis plan."
        )

        return self._fallback_plan()

    # ==================================================================
    # EDA CODE GENERATION
    # ==================================================================

    def generate_eda_code(
        self,
        plan: dict[str, Any],
        profile: dict[str, Any],
        filename: str | None = None,
    ) -> str:
        """
        Generate reproducible Python EDA code.

        The dataframe is already loaded by DataLens as `df`.

        The generated code:
          - does not reload the CSV/XLSX
          - does not hard-code the filename
          - does not use matplotlib
          - does not use seaborn
          - uses Pandas
          - uses Plotly
        """

        trimmed_profile = self._trim_profile(
            profile
        )

        base_user_message = (
            build_eda_code_user_message(
                plan,
                trimmed_profile,
                filename or "uploaded_data",
            )
        )

        guardrails = """
DATA LENS CODE GENERATION RULES

The DataLens application has already loaded the uploaded dataset into:

    df

Generate reproducible Python EDA code using this existing dataframe.

STRICT RULES:

1. Do NOT call pd.read_csv().

2. Do NOT call pd.read_excel().

3. Do NOT reload the dataset.

4. Do NOT hard-code sample_sales_data.csv.

5. Do NOT hard-code any other filename.

6. Do NOT invent columns.

7. Use only columns present in the supplied dataset profile.

8. Use Pandas for calculations.

9. Use Plotly for charts.

10. Do NOT use matplotlib.

11. Do NOT use seaborn.

12. Do NOT save PNG files.

13. Do NOT use plt.show().

14. Calculate KPI values directly from df.

15. Calculate grouped metrics directly from df.

16. Use actual dataframe values.

17. Do not fabricate numbers.

18. For Return_Flag encoded as 0/1, the mean represents the proportion
    of returned records. Multiply by 100 only when displaying it as a
    percentage.

19. Keep the generated code readable.

20. The code should assume that `df` already exists.

21. Return ONLY Python source code.

22. Do not return Markdown explanations.

23. Do not include ```python fences.

Example:

# df is already loaded by DataLens

total_revenue = df["Revenue"].sum()

print(
    f"Total Revenue: ${total_revenue:,.2f}"
)

revenue_by_region = (
    df.groupby(
        "Region",
        as_index=False
    )["Revenue"]
    .sum()
    .sort_values(
        "Revenue",
        ascending=False
    )
)

import plotly.express as px

fig = px.bar(
    revenue_by_region,
    x="Region",
    y="Revenue",
    title="Revenue by Region",
)

fig.show()

Adapt the code to the actual dataset and analysis plan.
"""

        user_msg = (
            f"{base_user_message}\n\n"
            f"{guardrails}"
        )

        def _call():
            return self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": EDA_CODE_SYSTEM,
                    },
                    {
                        "role": "user",
                        "content": user_msg,
                    },
                ],
                temperature=0.1,
            )

        response = self._call_with_retry(
            _call
        )

        raw_code = (
            response.choices[0].message.content
            if response.choices
            else ""
        ) or ""

        code = self._clean_code_response(
            raw_code
        )

        logger.debug(
            "Generated EDA code (%d chars)",
            len(code),
        )

        return code

    # ==================================================================
    # CODE CLEANING
    # ==================================================================

    @staticmethod
    def _clean_code_response(
        text: str,
    ) -> str:
        """
        Remove accidental Markdown fences around Python code.
        """

        if not text:
            return ""

        code = text.strip()

        if code.startswith(
            "```python"
        ):
            code = code[
                len("```python"):
            ].strip()

        elif code.startswith(
            "```py"
        ):
            code = code[
                len("```py"):
            ].strip()

        elif code.startswith(
            "```"
        ):
            code = code[
                3:
            ].strip()

        if code.endswith(
            "```"
        ):
            code = code[
                :-3
            ].strip()

        return code


# ======================================================================
# BACKWARD COMPATIBILITY
# ======================================================================

# Keeps older imports working if any existing project file still uses:
#
# from utils.groq_client import GrokClient
#
# The actual implementation is now GroqClient.

GrokClient = GroqClient
