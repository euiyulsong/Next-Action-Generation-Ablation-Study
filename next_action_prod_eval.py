#!/usr/bin/env python3
# next_action_prod_eval.py

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import textwrap

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel, Field


# ============================================================
# 0. CONFIG
# ============================================================

DEFAULT_MODEL = os.getenv("MODEL", "gpt-6-luna")
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-6-luna")

DEFAULT_REASONING = os.getenv(
    "REASONING_EFFORT",
    "none",
)

SEED = 42

MIN_PROMPT_CHARS = 20
MAX_PROMPT_CHARS = 1800
MAX_CONTEXT_CHARS = 3500


DATASET_PLAN = [
    ("dolly", 10),
    ("oasst1", 10),
    ("ultrachat", 10),
    ("ultrafeedback", 10),
    ("hh_rlhf", 10),
]


# ============================================================
# 1. ACTION TAXONOMY
# ============================================================

class FixedAction(str, Enum):
    NONE = "NONE"
    CLARIFY = "CLARIFY"
    EXPLAIN = "EXPLAIN"
    COMPARE = "COMPARE"
    SEARCH = "SEARCH"
    CODE = "CODE"
    TEST_OR_EVALUATE = "TEST_OR_EVALUATE"
    DEBUG = "DEBUG"
    CREATE = "CREATE"
    TOOL = "TOOL"
    PLAN = "PLAN"


# ============================================================
# 2. STRUCTURED OUTPUT SCHEMAS
# ============================================================

class JointFreeOutput(BaseModel):
    answer: str

    should_suggest: bool

    # 자유롭게 생성되는 internal metadata
    intent_tag: str = ""

    # 실제 사용자에게 보여줄 자연어 transition
    transition: str = ""

    rationale: str = ""


class FreeTransitionOutput(BaseModel):
    should_suggest: bool
    intent_tag: str = ""
    transition: str = ""
    rationale: str = ""


class FixedTransitionOutput(BaseModel):
    action: FixedAction
    transition: str = ""
    rationale: str = ""


class HybridTransitionOutput(BaseModel):
    action: FixedAction

    # coarse action은 fixed지만
    # 구체 행동은 free text
    action_detail: str = ""

    transition: str = ""
    rationale: str = ""


class GateOutput(BaseModel):
    should_suggest: bool

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    rationale: str


class GeneratedTransitionOutput(BaseModel):
    intent_tag: str
    transition: str


# ============================================================
# 3. JUDGE SCHEMA
# ============================================================

class JudgeItem(BaseModel):
    label: str

    # 실제 production에서 이 suggestion을 노출할 것인가
    should_show: bool

    relevance: int = Field(
        ge=1,
        le=5,
    )

    naturalness: int = Field(
        ge=1,
        le=5,
    )

    non_redundancy: int = Field(
        ge=1,
        le=5,
    )

    specificity: int = Field(
        ge=1,
        le=5,
    )

    usefulness: int = Field(
        ge=1,
        le=5,
    )

    # 1 = 문제 없음
    # 5 = 매우 심한 overreach
    overreach: int = Field(
        ge=1,
        le=5,
    )

    note: str


class JudgeBatch(BaseModel):
    # 이 sample 자체에 next step이 필요한가
    ideal_should_suggest: bool

    candidates: List[JudgeItem]


# ============================================================
# 4. PROMPTS
# ============================================================

ANSWER_SYSTEM = """
Answer the user's current request directly and completely.

Do not append generic follow-up offers such as:

- "Would you like me to..."
- "I can also..."
- "Let me know if you need anything else."

Do not add a next-step recommendation.

The answer must stand on its own.

Match the user's language when practical.
""".strip()


TRANSITION_POLICY = """
You are deciding whether to append a NATURAL conversational continuation
after an already-complete assistant answer.

The continuation is not a button label.
It is not an action enum shown to the user.

It should sound like the natural final sentence of a good assistant response.

Rules:

1. Prefer NO suggestion when the user's request is fully closed and there is
   no high-value continuation.

2. Suggest only something that materially advances the user's CURRENT goal.

3. Do not repeat work already completed in the answer.

4. Do not invent capabilities, external access, files, tools, or data.

5. Keep the continuation short:
   normally one sentence, at most two.

6. Match the user's language and conversational tone.

7. Avoid repetitive boilerplate such as:
   "Would you like me to..."
   when a more direct transition is natural.

8. A concrete continuation is better than a vague offer.

9. intent_tag is analytics metadata only.
   It should NOT constrain the wording.
   You may invent a new descriptive tag when appropriate.

10. If no continuation should be shown:

    should_suggest = false
    intent_tag = "NONE"
    transition = ""
""".strip()


FIXED_SPACE = """
Choose exactly one coarse internal action:

NONE
CLARIFY
EXPLAIN
COMPARE
SEARCH
CODE
TEST_OR_EVALUATE
DEBUG
CREATE
TOOL
PLAN

This label is INTERNAL metadata only.

The user-facing transition must still be natural free-form language.
""".strip()


# ------------------------------------------------------------
# JOINT
# ------------------------------------------------------------

JOINT_FREE_SYSTEM = f"""
Do two tasks inside ONE model generation.

TASK A:
Answer the user's request completely.

TASK B:
After forming the answer, decide whether a natural next-step continuation
belongs after that answer.

{TRANSITION_POLICY}

Important:

- Do not make the answer incomplete just to create a continuation.
- The answer must be useful even if the continuation is removed.
""".strip()


# ------------------------------------------------------------
# CONTINUATION FREE
# ------------------------------------------------------------

CONT_FREE_SYSTEM = f"""
{TRANSITION_POLICY}

You receive:

1. the original conversation
2. the COMPLETE assistant answer already given

Judge what is still useful AFTER reading that answer.
""".strip()


# ------------------------------------------------------------
# QUERY ONLY CONTROL
# ------------------------------------------------------------

QUERY_ONLY_SYSTEM = f"""
{TRANSITION_POLICY}

CONTROL CONDITION:

You do NOT see the answer that was generated.

Decide only from the user conversation.

This condition exists to measure whether answer-conditioning
actually improves next-step generation.
""".strip()


# ------------------------------------------------------------
# FIXED ACTION
# ------------------------------------------------------------

CONT_FIXED_SYSTEM = f"""
{TRANSITION_POLICY}

{FIXED_SPACE}

If action = NONE:

transition = ""
""".strip()


# ------------------------------------------------------------
# HYBRID
# ------------------------------------------------------------

CONT_HYBRID_SYSTEM = f"""
{TRANSITION_POLICY}

{FIXED_SPACE}

Also generate:

action_detail

This should describe the ACTUAL next action in free-form language.

Example:

action:
TEST_OR_EVALUATE

action_detail:
Compare reranker on/off using Recall@10 and MRR@10.

transition:
같은 validation set에서 reranker on/off ablation을 돌리면
실제 이득이 있는지 바로 확인할 수 있어.

The coarse action is only for logging/control.
It must NOT make the transition robotic.

If action = NONE:

action_detail = ""
transition = ""
""".strip()


# ------------------------------------------------------------
# TWO STAGE GATE
# ------------------------------------------------------------

GATE_SYSTEM = """
Decide ONLY whether a next-step continuation should be shown
after the completed answer.

Do NOT write the continuation.

Use a HIGH PRECISION threshold.

Return YES only when there is a clear, useful,
non-redundant continuation that materially advances
the user's current goal.

Return NO for:

- generic "more help"
- duplicated work
- topic drift
- merely possible follow-ups
- suggestions with little practical value
""".strip()


GENERATOR_SYSTEM = """
Generate ONE short, natural conversational continuation
after the provided complete answer.

Requirements:

- It must directly advance the user's current goal.
- It must not repeat the answer.
- It must not sound like a menu item.
- It must not sound like an action enum.
- It should usually be one sentence.
- Match the user's language and tone.
- intent_tag is free-form analytics metadata only.
""".strip()


# ============================================================
# 5. JUDGE PROMPT
# ============================================================

JUDGE_SYSTEM = """
You are evaluating NEXT-STEP CONTINUATIONS
for a production conversational assistant.

First decide whether an ideal assistant should show
ANY next-step continuation after the completed base answer.

Then score every candidate independently.

Scores are 1 to 5.

relevance:
Does it directly continue the user's current goal?

naturalness:
Does it sound like a natural conversational transition
after the answer?

non_redundancy:
Does it avoid repeating something already completed?

specificity:
Is it concrete rather than generic?

usefulness:
Would showing it materially help the user continue?

overreach:
1 = no overreach
5 = severe overreach, invented capability,
    needless action, or inappropriate expansion.

should_show:
Would you actually display THIS candidate
in a production assistant?

Do not reward verbosity.

A blank suggestion is correct when no next step is useful.
""".strip()


# ============================================================
# 6. TEXT UTILS
# ============================================================

def clean_text(x: Any) -> str:

    if x is None:
        return ""

    x = str(x)

    x = x.replace(
        "\x00",
        " ",
    )

    x = x.strip()

    x = re.sub(
        r"\s+",
        " ",
        x,
    )

    return x


def valid_prompt(text: str) -> bool:

    text = clean_text(text)

    return (
        MIN_PROMPT_CHARS
        <= len(text)
        <= MAX_PROMPT_CHARS
    )


def role_normalize(role: str) -> str:

    r = (
        role
        or ""
    ).lower()

    if r in {
        "user",
        "human",
        "prompter",
    }:
        return "user"

    return "assistant"


def conversation_to_text(
    messages: List[Dict[str, str]]
) -> str:

    parts = []

    for m in messages:

        role = (
            m.get(
                "role",
                "user",
            )
            .upper()
        )

        content = clean_text(
            m.get(
                "content",
                "",
            )
        )

        if content:
            parts.append(
                f"{role}: {content}"
            )

    result = "\n".join(
        parts
    )

    return result[
        -MAX_CONTEXT_CHARS:
    ]


def last_user_text(
    messages: List[Dict[str, str]]
) -> str:

    for m in reversed(messages):

        if (
            m.get("role")
            == "user"
        ):
            return clean_text(
                m.get(
                    "content",
                    "",
                )
            )

    return ""


# ============================================================
# 7. RESERVOIR SAMPLING
# ============================================================

def pick_rows(
    rows: Iterable[Dict[str, Any]],
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:

    rng = random.Random(
        seed
    )

    reservoir = []

    seen = 0

    # streaming dataset 전체를 다 훑지 않도록 제한
    cap = 25000

    for row in rows:

        seen += 1

        if len(reservoir) < n:

            reservoir.append(
                row
            )

        else:

            j = rng.randint(
                1,
                seen,
            )

            if j <= n:

                reservoir[
                    j - 1
                ] = row

        if seen >= cap:
            break

    return reservoir


# ============================================================
# 8. HH-RLHF PARSER
# ============================================================

def parse_hh_transcript(
    text: str
) -> List[Dict[str, str]]:

    if not text:
        return []

    pattern = re.compile(
        r"\n\n(Human|Assistant):\s*",
        re.MULTILINE,
    )

    raw = text

    if not raw.startswith(
        "\n\n"
    ):
        raw = (
            "\n\n"
            + raw
        )

    pieces = pattern.split(
        raw
    )

    msgs = []

    i = 1

    while (
        i + 1
        < len(pieces)
    ):

        role = role_normalize(
            pieces[i]
        )

        content = clean_text(
            pieces[
                i + 1
            ]
        )

        if content:

            msgs.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        i += 2

    return msgs


# ============================================================
# 9. GENERIC MESSAGE NORMALIZER
# ============================================================

def normalize_message_list(
    msgs: Any,
) -> List[Dict[str, str]]:

    out = []

    if not isinstance(
        msgs,
        list,
    ):
        return out

    for m in msgs:

        if not isinstance(
            m,
            dict,
        ):
            continue

        role = role_normalize(
            clean_text(
                m.get(
                    "role"
                )
            )
        )

        content = clean_text(
            m.get(
                "content"
            )
        )

        if content:

            out.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    return out


def truncate_to_last_user(
    messages: List[Dict[str, str]]
) -> List[Dict[str, str]]:

    user_positions = [
        i
        for i, m in enumerate(
            messages
        )
        if (
            m.get("role")
            == "user"
        )
    ]

    if not user_positions:
        return []

    end = user_positions[
        -1
    ]

    # 최근 6 message 정도만 사용
    clipped = messages[
        max(
            0,
            end - 5,
        ):
        end + 1
    ]

    if (
        not clipped
        or clipped[-1].get(
            "role"
        )
        != "user"
    ):
        return []

    return clipped


# ============================================================
# 10. DATASET: DOLLY
# ============================================================

def load_dolly(
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:

    ds = load_dataset(
        "databricks/databricks-dolly-15k",
        split="train",
        streaming=True,
    )

    def gen():

        for r in ds:

            instruction = clean_text(
                r.get(
                    "instruction"
                )
            )

            context = clean_text(
                r.get(
                    "context"
                )
            )

            if not valid_prompt(
                instruction
            ):
                continue

            user = instruction

            if context:

                user += (
                    "\n\nContext:\n"
                    + context[:1200]
                )

            if (
                len(user)
                > MAX_PROMPT_CHARS
            ):
                continue

            yield {
                "source": "dolly",

                "source_category":
                    clean_text(
                        r.get(
                            "category"
                        )
                    ),

                "messages": [
                    {
                        "role": "user",
                        "content": user,
                    }
                ],

                "user_text": user,
            }

    return pick_rows(
        gen(),
        n,
        seed + 11,
    )


# ============================================================
# 11. DATASET: OPENASSISTANT
# ============================================================

def load_oasst1(
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:

    ds = load_dataset(
        "OpenAssistant/oasst1",
        split="train",
        streaming=True,
    )

    def gen():

        for r in ds:

            role = clean_text(
                r.get(
                    "role"
                )
            ).lower()

            if role != "prompter":
                continue

            if (
                r.get(
                    "parent_id"
                )
                not in (
                    None,
                    "",
                )
            ):
                continue

            lang = clean_text(
                r.get(
                    "lang"
                )
            ).lower()

            if lang not in {
                "en",
                "eng",
            }:
                continue

            text = clean_text(
                r.get(
                    "text"
                )
            )

            if not valid_prompt(
                text
            ):
                continue

            yield {
                "source": "oasst1",

                "source_category":
                    "root_prompt",

                "messages": [
                    {
                        "role": "user",
                        "content": text,
                    }
                ],

                "user_text": text,
            }

    return pick_rows(
        gen(),
        n,
        seed + 22,
    )


# ============================================================
# 12. DATASET: ULTRACHAT
# ============================================================

def load_ultrachat(
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:

    last_err = None

    ds = None

    for split in (
        "train_sft",
        "train_gen",
    ):

        try:

            ds = load_dataset(
                "HuggingFaceH4/ultrachat_200k",
                split=split,
                streaming=True,
            )

            break

        except Exception as e:

            last_err = e

    if ds is None:

        raise RuntimeError(
            "Could not load "
            f"UltraChat: {last_err}"
        )

    def gen():

        for r in ds:

            msgs = normalize_message_list(
                r.get(
                    "messages"
                )
            )

            msgs = truncate_to_last_user(
                msgs
            )

            if not msgs:
                continue

            user = last_user_text(
                msgs
            )

            context = conversation_to_text(
                msgs
            )

            if not valid_prompt(
                user
            ):
                continue

            if (
                len(context)
                > MAX_CONTEXT_CHARS
            ):
                continue

            yield {
                "source":
                    "ultrachat",

                "source_category":
                    (
                        "multi_turn"
                        if len(msgs) > 1
                        else "single_turn"
                    ),

                "messages":
                    msgs,

                "user_text":
                    user,
            }

    return pick_rows(
        gen(),
        n,
        seed + 33,
    )


# ============================================================
# 13. DATASET: ULTRAFEEDBACK
# ============================================================

def load_ultrafeedback(
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:

    ds = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",
        streaming=True,
    )

    def gen():

        for r in ds:

            msgs = normalize_message_list(
                r.get(
                    "messages"
                )
            )

            msgs = truncate_to_last_user(
                msgs
            )

            if not msgs:

                p = clean_text(
                    r.get(
                        "prompt"
                    )
                )

                if not valid_prompt(
                    p
                ):
                    continue

                msgs = [
                    {
                        "role": "user",
                        "content": p,
                    }
                ]

            user = last_user_text(
                msgs
            )

            if not valid_prompt(
                user
            ):
                continue

            yield {
                "source":
                    "ultrafeedback",

                "source_category":
                    "preference_prompt",

                "messages":
                    msgs,

                "user_text":
                    user,
            }

    return pick_rows(
        gen(),
        n,
        seed + 44,
    )


# ============================================================
# 14. DATASET: ANTHROPIC HH-RLHF
# ============================================================

def load_hh_rlhf(
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:

    attempts = [
        {
            "path":
                "Anthropic/hh-rlhf",

            "data_dir":
                "helpful-base",

            "split":
                "train",
        },

        {
            "path":
                "Anthropic/hh-rlhf",

            "name":
                "helpful-base",

            "split":
                "train",
        },
    ]

    ds = None
    last_err = None

    for kwargs in attempts:

        try:

            ds = load_dataset(
                streaming=True,
                **kwargs,
            )

            break

        except Exception as e:

            last_err = e

    if ds is None:

        raise RuntimeError(
            "Could not load HH-RLHF: "
            f"{last_err}"
        )

    def gen():

        for r in ds:

            chosen = clean_text(
                r.get(
                    "chosen"
                )
            )

            msgs = parse_hh_transcript(
                chosen
            )

            msgs = truncate_to_last_user(
                msgs
            )

            if not msgs:
                continue

            user = last_user_text(
                msgs
            )

            if not valid_prompt(
                user
            ):
                continue

            yield {
                "source":
                    "hh_rlhf",

                "source_category":
                    "helpful_base",

                "messages":
                    msgs,

                "user_text":
                    user,
            }

    return pick_rows(
        gen(),
        n,
        seed + 55,
    )


# ============================================================
# 15. DATASET REGISTRY
# ============================================================

LOADERS = {
    "dolly":
        load_dolly,

    "oasst1":
        load_oasst1,

    "ultrachat":
        load_ultrachat,

    "ultrafeedback":
        load_ultrafeedback,

    "hh_rlhf":
        load_hh_rlhf,
}


# ============================================================
# 16. BUILD 50 SAMPLES
# ============================================================

def build_samples(
    total_n: int,
    seed: int,
) -> List[Dict[str, Any]]:

    if total_n <= 0:

        raise ValueError(
            "--n must be > 0"
        )

    sources = [
        x[0]
        for x in DATASET_PLAN
    ]

    base = (
        total_n
        // len(sources)
    )

    rem = (
        total_n
        % len(sources)
    )

    plan = [
        (
            source,

            base
            + (
                1
                if i < rem
                else 0
            ),
        )

        for i, source
        in enumerate(
            sources
        )
    ]

    samples = []

    for source, count in plan:

        print(
            f"[dataset] "
            f"{source}: "
            f"sampling {count}"
        )

        try:

            rows = LOADERS[
                source
            ](
                count,
                seed,
            )

        except Exception as e:

            print(
                f"[dataset] WARNING "
                f"{source} failed: "
                f"{e}"
            )

            rows = []

        if len(rows) < count:

            print(
                f"[dataset] WARNING "
                f"{source}: "
                f"{len(rows)}/{count}"
            )

        samples.extend(
            rows
        )

    # --------------------------------------------------------
    # Backfill
    # --------------------------------------------------------

    if (
        len(samples)
        < total_n
    ):

        need = (
            total_n
            - len(samples)
        )

        print(
            f"[dataset] backfilling "
            f"{need} from Dolly"
        )

        extra = load_dolly(
            need * 2,
            seed + 999,
        )

        existing = {
            clean_text(
                x[
                    "user_text"
                ]
            )
            for x in samples
        }

        for x in extra:

            key = clean_text(
                x[
                    "user_text"
                ]
            )

            if key in existing:
                continue

            x[
                "source"
            ] = (
                x["source"]
                + "_backfill"
            )

            samples.append(
                x
            )

            existing.add(
                key
            )

            if (
                len(samples)
                >= total_n
            ):
                break

    samples = samples[
        :total_n
    ]

    rng = random.Random(
        seed
    )

    rng.shuffle(
        samples
    )

    for i, s in enumerate(
        samples,
        1,
    ):

        s[
            "sample_id"
        ] = i

    return samples


# ============================================================
# 17. OPENAI RUNNER
# ============================================================

class Runner:

    def __init__(
        self,
        model: str,
        reasoning: str,
    ):

        self.client = OpenAI()

        self.model = model

        self.reasoning = reasoning


    def _input_with_system(
        self,
        system: str,
        messages: List[Dict[str, str]],
    ):

        return [
            {
                "role":
                    "system",

                "content":
                    system,
            }
        ] + messages


    # --------------------------------------------------------
    # BASE ANSWER
    # --------------------------------------------------------

    def answer(
        self,
        messages: List[Dict[str, str]],
    ) -> str:

        r = self.client.responses.create(

            model=
                self.model,

            reasoning={
                "effort":
                    self.reasoning
            },

            input=
                self._input_with_system(
                    ANSWER_SYSTEM,
                    messages,
                ),
        )

        return (
            r.output_text
            .strip()
        )


    # --------------------------------------------------------
    # JOINT FREE
    # --------------------------------------------------------

    def joint_free(
        self,
        messages: List[Dict[str, str]],
    ) -> JointFreeOutput:

        r = self.client.responses.parse(

            model=
                self.model,

            reasoning={
                "effort":
                    self.reasoning
            },

            input=
                self._input_with_system(
                    JOINT_FREE_SYSTEM,
                    messages,
                ),

            text_format=
                JointFreeOutput,
        )

        return r.output_parsed


    # --------------------------------------------------------
    # Continuation helper
    # --------------------------------------------------------

    def _continuation_input(
        self,
        system: str,
        messages: List[Dict[str, str]],
        answer: Optional[str],
    ):

        inp = (
            self._input_with_system(
                system,
                messages,
            )
        )

        if answer is not None:

            inp += [
                {
                    "role":
                        "assistant",

                    "content":
                        answer,
                },

                {
                    "role":
                        "user",

                    "content":
                        (
                            "[Evaluation instruction: "
                            "decide only whether/how to "
                            "append a next-step continuation "
                            "after the assistant answer above.]"
                        ),
                },
            ]

        return inp


    # --------------------------------------------------------
    # FREE CONTINUATION
    # --------------------------------------------------------

    def cont_free(
        self,
        messages: List[Dict[str, str]],
        answer: str,
    ) -> FreeTransitionOutput:

        r = self.client.responses.parse(

            model=
                self.model,

            reasoning={
                "effort":
                    self.reasoning
            },

            input=
                self._continuation_input(
                    CONT_FREE_SYSTEM,
                    messages,
                    answer,
                ),

            text_format=
                FreeTransitionOutput,
        )

        return r.output_parsed


    # --------------------------------------------------------
    # QUERY ONLY CONTROL
    # --------------------------------------------------------

    def query_only_free(
        self,
        messages: List[Dict[str, str]],
    ) -> FreeTransitionOutput:

        r = self.client.responses.parse(

            model=
                self.model,

            reasoning={
                "effort":
                    self.reasoning
            },

            input=
                self._continuation_input(
                    QUERY_ONLY_SYSTEM,
                    messages,
                    None,
                ),

            text_format=
                FreeTransitionOutput,
        )

        return r.output_parsed


    # --------------------------------------------------------
    # FIXED
    # --------------------------------------------------------

    def cont_fixed(
        self,
        messages: List[Dict[str, str]],
        answer: str,
    ) -> FixedTransitionOutput:

        r = self.client.responses.parse(

            model=
                self.model,

            reasoning={
                "effort":
                    self.reasoning
            },

            input=
                self._continuation_input(
                    CONT_FIXED_SYSTEM,
                    messages,
                    answer,
                ),

            text_format=
                FixedTransitionOutput,
        )

        return r.output_parsed


    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

    def cont_hybrid(
        self,
        messages: List[Dict[str, str]],
        answer: str,
    ) -> HybridTransitionOutput:

        r = self.client.responses.parse(

            model=
                self.model,

            reasoning={
                "effort":
                    self.reasoning
            },

            input=
                self._continuation_input(
                    CONT_HYBRID_SYSTEM,
                    messages,
                    answer,
                ),

            text_format=
                HybridTransitionOutput,
        )

        return r.output_parsed


    # --------------------------------------------------------
    # GATE
    # --------------------------------------------------------

    def gate(
        self,
        messages: List[Dict[str, str]],
        answer: str,
    ) -> GateOutput:

        r = self.client.responses.parse(

            model=
                self.model,

            reasoning={
                "effort":
                    self.reasoning
            },

            input=
                self._continuation_input(
                    GATE_SYSTEM,
                    messages,
                    answer,
                ),

            text_format=
                GateOutput,
        )

        return r.output_parsed


    # --------------------------------------------------------
    # FREE GENERATOR
    # --------------------------------------------------------

    def generate_transition(
        self,
        messages: List[Dict[str, str]],
        answer: str,
    ) -> GeneratedTransitionOutput:

        r = self.client.responses.parse(

            model=
                self.model,

            reasoning={
                "effort":
                    self.reasoning
            },

            input=
                self._continuation_input(
                    GENERATOR_SYSTEM,
                    messages,
                    answer,
                ),

            text_format=
                GeneratedTransitionOutput,
        )

        return r.output_parsed


# ============================================================
# 18. EXPERIMENT METHODS
# ============================================================

METHODS = [
    "joint_free",
    "cont_free",
    "query_only_free",
    "cont_fixed",
    "cont_hybrid",
    "two_stage_free",
]


def run_one(
    runner: Runner,
    sample: Dict[str, Any],
) -> Tuple[
    str,
    List[Dict[str, Any]],
]:

    msgs = sample[
        "messages"
    ]

    # ========================================================
    # Shared answer
    # ========================================================

    base_answer = runner.answer(
        msgs
    )

    results = []


    # ========================================================
    # 1. JOINT FREE
    # ========================================================

    t0 = time.perf_counter()

    jf = runner.joint_free(
        msgs
    )

    latency = (
        time.perf_counter()
        - t0
    )

    results.append(
        {
            "method":
                "joint_free",

            "answer":
                jf.answer,

            "should_suggest":
                bool(
                    jf.should_suggest
                ),

            "action_meta":
                (
                    clean_text(
                        jf.intent_tag
                    )
                    or (
                        "NONE"
                        if not jf.should_suggest
                        else ""
                    )
                ),

            "transition":
                (
                    clean_text(
                        jf.transition
                    )
                    if jf.should_suggest
                    else ""
                ),

            "rationale":
                clean_text(
                    jf.rationale
                ),

            "latency_s":
                latency,

            # main answer와 함께 생성되므로
            # answer 이후 추가 call은 0
            "extra_calls_after_answer":
                0,
        }
    )


    # ========================================================
    # 2. CONTINUATION FREE
    # ========================================================

    t0 = time.perf_counter()

    cf = runner.cont_free(
        msgs,
        base_answer,
    )

    latency = (
        time.perf_counter()
        - t0
    )

    results.append(
        {
            "method":
                "cont_free",

            "answer":
                base_answer,

            "should_suggest":
                bool(
                    cf.should_suggest
                ),

            "action_meta":
                (
                    clean_text(
                        cf.intent_tag
                    )
                    or (
                        "NONE"
                        if not cf.should_suggest
                        else ""
                    )
                ),

            "transition":
                (
                    clean_text(
                        cf.transition
                    )
                    if cf.should_suggest
                    else ""
                ),

            "rationale":
                clean_text(
                    cf.rationale
                ),

            "latency_s":
                latency,

            "extra_calls_after_answer":
                1,
        }
    )


    # ========================================================
    # 3. QUERY ONLY FREE
    # ========================================================

    t0 = time.perf_counter()

    qf = runner.query_only_free(
        msgs
    )

    latency = (
        time.perf_counter()
        - t0
    )

    results.append(
        {
            "method":
                "query_only_free",

            "answer":
                base_answer,

            "should_suggest":
                bool(
                    qf.should_suggest
                ),

            "action_meta":
                (
                    clean_text(
                        qf.intent_tag
                    )
                    or (
                        "NONE"
                        if not qf.should_suggest
                        else ""
                    )
                ),

            "transition":
                (
                    clean_text(
                        qf.transition
                    )
                    if qf.should_suggest
                    else ""
                ),

            "rationale":
                clean_text(
                    qf.rationale
                ),

            "latency_s":
                latency,

            "extra_calls_after_answer":
                1,
        }
    )


    # ========================================================
    # 4. FIXED CONTINUATION
    # ========================================================

    t0 = time.perf_counter()

    fx = runner.cont_fixed(
        msgs,
        base_answer,
    )

    latency = (
        time.perf_counter()
        - t0
    )

    results.append(
        {
            "method":
                "cont_fixed",

            "answer":
                base_answer,

            "should_suggest":
                (
                    fx.action
                    != FixedAction.NONE
                ),

            "action_meta":
                fx.action.value,

            "transition":
                (
                    clean_text(
                        fx.transition
                    )
                    if (
                        fx.action
                        != FixedAction.NONE
                    )
                    else ""
                ),

            "rationale":
                clean_text(
                    fx.rationale
                ),

            "latency_s":
                latency,

            "extra_calls_after_answer":
                1,
        }
    )


    # ========================================================
    # 5. HYBRID CONTINUATION
    # ========================================================

    t0 = time.perf_counter()

    hy = runner.cont_hybrid(
        msgs,
        base_answer,
    )

    latency = (
        time.perf_counter()
        - t0
    )

    detail = clean_text(
        hy.action_detail
    )

    if detail:

        meta = (
            f"{hy.action.value}: "
            f"{detail}"
        )

    else:

        meta = (
            hy.action.value
        )

    results.append(
        {
            "method":
                "cont_hybrid",

            "answer":
                base_answer,

            "should_suggest":
                (
                    hy.action
                    != FixedAction.NONE
                ),

            "action_meta":
                meta,

            "transition":
                (
                    clean_text(
                        hy.transition
                    )
                    if (
                        hy.action
                        != FixedAction.NONE
                    )
                    else ""
                ),

            "rationale":
                clean_text(
                    hy.rationale
                ),

            "latency_s":
                latency,

            "extra_calls_after_answer":
                1,
        }
    )


    # ========================================================
    # 6. TWO STAGE FREE
    # ========================================================

    t0 = time.perf_counter()

    gate = runner.gate(
        msgs,
        base_answer,
    )

    transition = ""

    intent_tag = "NONE"

    calls = 1

    if gate.should_suggest:

        gt = (
            runner.generate_transition(
                msgs,
                base_answer,
            )
        )

        transition = clean_text(
            gt.transition
        )

        intent_tag = (
            clean_text(
                gt.intent_tag
            )
            or "FREE_ACTION"
        )

        calls = 2

    latency = (
        time.perf_counter()
        - t0
    )

    results.append(
        {
            "method":
                "two_stage_free",

            "answer":
                base_answer,

            "should_suggest":
                bool(
                    gate.should_suggest
                ),

            "action_meta":
                intent_tag,

            "transition":
                transition,

            "rationale":
                clean_text(
                    gate.rationale
                ),

            "latency_s":
                latency,

            "extra_calls_after_answer":
                calls,

            "gate_confidence":
                gate.confidence,
        }
    )

    return (
        base_answer,
        results,
    )


# ============================================================
# 19. LLM JUDGE
# ============================================================

def judge_sample(
    client: OpenAI,
    judge_model: str,
    reasoning: str,
    sample: Dict[str, Any],
    base_answer: str,
    method_rows: List[Dict[str, Any]],
    seed: int,
) -> List[Dict[str, Any]]:

    rng = random.Random(
        seed
        + int(
            sample[
                "sample_id"
            ]
        )
    )

    shuffled = list(
        method_rows
    )

    rng.shuffle(
        shuffled
    )

    labels = [
        chr(
            ord("A")
            + i
        )
        for i
        in range(
            len(shuffled)
        )
    ]

    label_to_method = {}

    candidate_text = []

    for label, row in zip(
        labels,
        shuffled,
    ):

        label_to_method[
            label
        ] = row[
            "method"
        ]

        if row[
            "should_suggest"
        ]:

            transition = row[
                "transition"
            ]

        else:

            transition = (
                "[NO SUGGESTION]"
            )

        candidate_text.append(
            f"{label}: "
            f"{transition}"
        )

    prompt = f"""
CONVERSATION:
{conversation_to_text(sample["messages"])}

COMPLETED BASE ANSWER:
{base_answer}

CANDIDATES:
{chr(10).join(candidate_text)}
""".strip()

    r = client.responses.parse(

        model=
            judge_model,

        reasoning={
            "effort":
                reasoning
        },

        input=[
            {
                "role":
                    "system",

                "content":
                    JUDGE_SYSTEM,
            },

            {
                "role":
                    "user",

                "content":
                    prompt,
            },
        ],

        text_format=
            JudgeBatch,
    )

    parsed = (
        r.output_parsed
    )

    by_label = {
        x.label:
            x

        for x in
        parsed.candidates
    }

    out = []

    for label in labels:

        if label not in by_label:
            continue

        x = by_label[
            label
        ]

        out.append(
            {
                "sample_id":
                    sample[
                        "sample_id"
                    ],

                "source":
                    sample[
                        "source"
                    ],

                "method":
                    label_to_method[
                        label
                    ],

                "ideal_should_suggest":
                    parsed.ideal_should_suggest,

                "judge_should_show":
                    x.should_show,

                "relevance":
                    x.relevance,

                "naturalness":
                    x.naturalness,

                "non_redundancy":
                    x.non_redundancy,

                "specificity":
                    x.specificity,

                "usefulness":
                    x.usefulness,

                "overreach":
                    x.overreach,

                "judge_note":
                    x.note,
            }
        )

    return out


# ============================================================
# 20. PNG HELPERS
# ============================================================

def wrap(
    s: str,
    width: int = 100,
) -> str:

    if not s:
        return ""

    return "\n".join(
        textwrap.wrap(
            str(s),
            width=width,
            replace_whitespace=False,
        )
    )


# ============================================================
# 21. SAMPLE CARD PNG
# ============================================================

def save_sample_card(
    sample: Dict[str, Any],
    rows: List[Dict[str, Any]],
    out_dir: Path,
):

    fig = plt.figure(
        figsize=(
            18,
            14,
        )
    )

    plt.axis(
        "off"
    )

    parts = [
        (
            f"SAMPLE "
            f"#{sample['sample_id']:02d}"
            f" | source="
            f"{sample['source']}"
            f" | category="
            f"{sample.get('source_category','')}"
        ),

        "",

        "CONVERSATION",

        wrap(
            conversation_to_text(
                sample[
                    "messages"
                ]
            ),
            110,
        ),

        "",

        "=" * 110,
    ]

    for row in rows:

        transition = (
            row.get(
                "transition",
                ""
            )
            if row.get(
                "should_suggest",
                False,
            )
            else "[NO SUGGESTION]"
        )

        parts += [
            "",

            (
                f"[{row['method']}]"
                f"  show="
                f"{row['should_suggest']}"
                f"  meta="
                f"{row.get('action_meta','')}"
                f"  latency="
                f"{row.get('latency_s',0):.2f}s"
            ),

            (
                "Transition: "
                + wrap(
                    transition,
                    105,
                )
            ),
        ]

    plt.text(
        0.01,
        0.99,
        "\n".join(
            parts
        ),
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.7,
    )

    plt.tight_layout()

    plt.savefig(
        out_dir
        / (
            f"sample_"
            f"{sample['sample_id']:02d}.png"
        ),
        dpi=155,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 22. OVERVIEW PNG
# ============================================================

def save_overviews(
    samples: List[Dict[str, Any]],
    rows_by_sample: Dict[
        int,
        List[Dict[str, Any]]
    ],
    out_dir: Path,
):

    page_size = 5

    for start in range(
        0,
        len(samples),
        page_size,
    ):

        chunk = samples[
            start:
            start + page_size
        ]

        fig, axes = plt.subplots(
            len(chunk),
            1,
            figsize=(
                20,
                5.3 * len(chunk),
            ),
        )

        if len(chunk) == 1:

            axes = [
                axes
            ]

        for ax, sample in zip(
            axes,
            chunk,
        ):

            ax.axis(
                "off"
            )

            rows = rows_by_sample[
                sample[
                    "sample_id"
                ]
            ]

            text = [
                (
                    f"#{sample['sample_id']:02d}"
                    f" [{sample['source']}] "
                    f"{wrap(sample['user_text'], 105)}"
                ),

                "",
            ]

            for row in rows:

                if row[
                    "should_suggest"
                ]:

                    transition = row[
                        "transition"
                    ]

                else:

                    transition = "[NONE]"

                text.append(
                    (
                        f"{row['method']:16s}"
                        f" | "
                        f"{wrap(transition, 105)}"
                    )
                )

            ax.text(
                0.01,
                0.98,
                "\n".join(
                    text
                ),
                va="top",
                ha="left",
                family="monospace",
                fontsize=8.5,
            )

        plt.tight_layout()

        lo = chunk[
            0
        ][
            "sample_id"
        ]

        hi = chunk[
            -1
        ][
            "sample_id"
        ]

        plt.savefig(
            out_dir
            / (
                f"overview_"
                f"{lo:02d}_"
                f"{hi:02d}.png"
            ),
            dpi=155,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# 23. SUMMARY PLOTS
# ============================================================

def save_summary_plots(
    df: pd.DataFrame,
    out_dir: Path,
    judge_df: Optional[
        pd.DataFrame
    ] = None,
):

    order = METHODS

    # --------------------------------------------------------
    # Suggestion Rate
    # --------------------------------------------------------

    rates = [
        df.loc[
            df.method == m,
            "should_suggest",
        ]
        .astype(float)
        .mean()

        for m in order
    ]

    plt.figure(
        figsize=(
            11,
            5,
        )
    )

    plt.bar(
        order,
        rates,
    )

    plt.ylim(
        0,
        1,
    )

    plt.ylabel(
        "Suggestion rate"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.title(
        "How often each method shows "
        "a next-step continuation"
    )

    plt.tight_layout()

    plt.savefig(
        out_dir
        / "suggestion_rate.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


    # --------------------------------------------------------
    # Judge scores
    # --------------------------------------------------------

    if (
        judge_df is not None
        and not judge_df.empty
    ):

        metric_cols = [
            "relevance",
            "naturalness",
            "non_redundancy",
            "specificity",
            "usefulness",
        ]

        means = (
            judge_df
            .groupby(
                "method"
            )[
                metric_cols
            ]
            .mean()
            .reindex(
                order
            )
        )

        ax = means.plot(
            kind="bar",
            figsize=(
                13,
                6,
            ),
        )

        ax.set_ylim(
            1,
            5,
        )

        ax.set_ylabel(
            "Judge mean score (1-5)"
        )

        ax.set_title(
            "Next-step quality by method"
        )

        plt.xticks(
            rotation=30,
            ha="right",
        )

        plt.tight_layout()

        plt.savefig(
            out_dir
            / "judge_scores.png",
            dpi=180,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# 24. HUMAN ANNOTATION CSV
# ============================================================

def make_human_annotation(
    df: pd.DataFrame,
    out_path: Path,
):

    cols = [
        "sample_id",
        "source",
        "source_category",
        "conversation",
        "base_answer",
    ]

    wide_base = (
        df[
            cols
        ]
        .drop_duplicates(
            "sample_id"
        )
        .set_index(
            "sample_id"
        )
    )

    for method in METHODS:

        sub = (
            df[
                df.method
                == method
            ]
            .set_index(
                "sample_id"
            )
        )

        wide_base[
            f"{method}__show"
        ] = sub[
            "should_suggest"
        ]

        wide_base[
            f"{method}__transition"
        ] = sub[
            "transition"
        ]

    ann = (
        wide_base
        .reset_index()
    )

    ann[
        "gold_should_suggest"
    ] = ""

    ann[
        "best_method"
    ] = ""

    ann[
        "best_timing"
    ] = ""

    ann[
        "best_action_policy"
    ] = ""

    ann[
        "notes"
    ] = ""

    for method in METHODS:

        ann[
            f"{method}__human_score"
        ] = ""

    ann.to_csv(
        out_path,
        index=False,
    )


# ============================================================
# 25. SUMMARY TABLE
# ============================================================

def make_summary(
    df: pd.DataFrame,
    judge_df: Optional[
        pd.DataFrame
    ],
) -> pd.DataFrame:

    rows = []

    for method in METHODS:

        x = df[
            df.method
            == method
        ]

        row = {
            "method":
                method,

            "n":
                len(x),

            "suggestion_rate":
                (
                    x[
                        "should_suggest"
                    ]
                    .astype(float)
                    .mean()
                ),

            "mean_latency_s_action_stage":
                x[
                    "latency_s"
                ].mean(),

            "mean_extra_calls_after_answer":
                x[
                    "extra_calls_after_answer"
                ].mean(),

            "unique_action_meta":
                (
                    x[
                        "action_meta"
                    ]
                    .replace(
                        "",
                        "NONE",
                    )
                    .nunique()
                ),
        }

        if (
            judge_df
            is not None
            and not judge_df.empty
        ):

            j = judge_df[
                judge_df.method
                == method
            ]

            for c in [
                "relevance",
                "naturalness",
                "non_redundancy",
                "specificity",
                "usefulness",
                "overreach",
            ]:

                row[
                    f"judge_{c}"
                ] = (
                    j[
                        c
                    ].mean()
                )

            row[
                "judge_show_rate"
            ] = (
                j[
                    "judge_should_show"
                ]
                .astype(float)
                .mean()
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 26. PRODUCTION DECISION REPORT
# ============================================================

def write_production_readout(
    summary: pd.DataFrame,
    out_path: Path,
):

    lines = [
        "# Production decision guide",
        "",

        "## 1. JOINT_FREE vs CONT_FREE",
        "",

        (
            "- If `cont_free` improves naturalness "
            "and non-redundancy enough to justify "
            "extra latency, prefer answer-conditioned continuation."
        ),

        (
            "- If quality is nearly tied, "
            "`joint_free` is operationally simpler."
        ),

        "",

        "## 2. CONT_FREE vs TWO_STAGE_FREE",
        "",

        (
            "- `two_stage_free` is useful only if the "
            "dedicated gate materially reduces unwanted suggestions."
        ),

        (
            "- If suppression quality is similar, "
            "`cont_free` is the cheaper production choice."
        ),

        "",

        "## 3. QUERY_ONLY_FREE",
        "",

        (
            "- If it trails `cont_free`, the completed answer "
            "contains useful information for deciding what remains."
        ),

        (
            "- If it is similar to `cont_free`, next-action generation "
            "may be parallelized with answer generation."
        ),

        "",

        "## 4. FIXED / HYBRID",
        "",

        (
            "- Use a coarse fixed action when needed for "
            "tool permissions, routing, analytics, or deterministic UI."
        ),

        (
            "- For purely conversational follow-ups, "
            "free-form transition generation may be preferable."
        ),

        "",

        "## 5. Metrics priority",
        "",

        "1. should-show precision",
        "2. naturalness",
        "3. non-redundancy",
        "4. usefulness",
        "5. latency / extra calls",

        "",

        "## Raw summary",
        "",

        summary.to_markdown(
            index=False
        ),
        "",
    ]

    out_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# ============================================================
# 27. MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--n",
        type=int,
        default=50,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    ap.add_argument(
        "--model",
        default=
            DEFAULT_MODEL,
    )

    ap.add_argument(
        "--reasoning",
        default=
            DEFAULT_REASONING,
    )

    ap.add_argument(
        "--judge",
        action="store_true",
    )

    ap.add_argument(
        "--judge-model",
        default=
            DEFAULT_JUDGE_MODEL,
    )

    ap.add_argument(
        "--out",
        default=
            "next_action_prod_results",
    )

    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
    )

    args = ap.parse_args()


    # ========================================================
    # Output dirs
    # ========================================================

    out_dir = Path(
        args.out
    )

    cards_dir = (
        out_dir
        / "cards"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cards_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    print(
        f"[config] model="
        f"{args.model}, "
        f"reasoning="
        f"{args.reasoning}, "
        f"n={args.n}"
    )


    # ========================================================
    # Dataset
    # ========================================================

    samples = build_samples(
        args.n,
        args.seed,
    )

    with open(
        out_dir
        / "samples.jsonl",
        "w",
        encoding="utf-8",
    ) as f:

        for s in samples:

            f.write(
                json.dumps(
                    s,
                    ensure_ascii=False,
                )
                + "\n"
            )


    # ========================================================
    # Run experiment
    # ========================================================

    runner = Runner(
        args.model,
        args.reasoning,
    )

    all_rows = []

    rows_by_sample = {}

    base_answers = {}


    for i, sample in enumerate(
        samples,
        1,
    ):

        print(
            "\n"
            f"[{i:02d}/"
            f"{len(samples)}] "
            f"{sample['source']} | "
            f"{sample['user_text'][:120]}"
        )

        try:

            (
                base_answer,
                method_rows,
            ) = run_one(
                runner,
                sample,
            )

        except Exception as e:

            print(
                f"[ERROR] sample "
                f"{sample['sample_id']}: "
                f"{e}"
            )

            continue


        base_answers[
            sample[
                "sample_id"
            ]
        ] = base_answer


        rows_by_sample[
            sample[
                "sample_id"
            ]
        ] = method_rows


        for row in method_rows:

            row.update(
                {
                    "sample_id":
                        sample[
                            "sample_id"
                        ],

                    "source":
                        sample[
                            "source"
                        ],

                    "source_category":
                        sample.get(
                            "source_category",
                            "",
                        ),

                    "user_text":
                        sample[
                            "user_text"
                        ],

                    "conversation":
                        conversation_to_text(
                            sample[
                                "messages"
                            ]
                        ),

                    "base_answer":
                        base_answer,
                }
            )

            all_rows.append(
                row
            )


        # incremental save
        pd.DataFrame(
            all_rows
        ).to_csv(
            out_dir
            / "results_partial.csv",
            index=False,
        )


        save_sample_card(
            sample,
            method_rows,
            cards_dir,
        )


        if args.sleep:

            time.sleep(
                args.sleep
            )


    if not all_rows:

        raise RuntimeError(
            "No experiment rows "
            "were produced."
        )


    # ========================================================
    # Long-format CSV
    # ========================================================

    df = pd.DataFrame(
        all_rows
    )

    df.to_csv(
        out_dir
        / "results_long.csv",
        index=False,
    )


    # ========================================================
    # Wide-format CSV
    # ========================================================

    wide = df.pivot_table(

        index=[
            "sample_id",
            "source",
            "source_category",
            "user_text",
            "conversation",
            "base_answer",
        ],

        columns=
            "method",

        values=[
            "should_suggest",
            "action_meta",
            "transition",
            "rationale",
            "latency_s",
        ],

        aggfunc=
            "first",
    )


    wide.columns = [
        f"{a}__{b}"

        for a, b
        in wide.columns
    ]


    wide.reset_index().to_csv(
        out_dir
        / "results_wide.csv",
        index=False,
    )


    # ========================================================
    # Optional judge
    # ========================================================

    judge_df = None

    if args.judge:

        judge_client = OpenAI()

        judged = []

        valid_samples = [
            s
            for s in samples
            if (
                s[
                    "sample_id"
                ]
                in rows_by_sample
            )
        ]

        for k, sample in enumerate(
            valid_samples,
            1,
        ):

            print(
                f"[judge "
                f"{k:02d}/"
                f"{len(valid_samples)}] "
                f"sample="
                f"{sample['sample_id']}"
            )

            try:

                items = judge_sample(
                    judge_client,
                    args.judge_model,
                    args.reasoning,
                    sample,
                    base_answers[
                        sample[
                            "sample_id"
                        ]
                    ],
                    rows_by_sample[
                        sample[
                            "sample_id"
                        ]
                    ],
                    args.seed,
                )

                judged.extend(
                    items
                )

            except Exception as e:

                print(
                    f"[judge ERROR] "
                    f"sample "
                    f"{sample['sample_id']}: "
                    f"{e}"
                )


        judge_df = pd.DataFrame(
            judged
        )

        if not judge_df.empty:

            judge_df.to_csv(
                out_dir
                / "judge_scores.csv",
                index=False,
            )


    # ========================================================
    # Intent fragmentation
    # ========================================================

    frag = (
        df
        .groupby(
            "method"
        )
        .agg(
            n=(
                "sample_id",
                "count",
            ),

            suggestions=(
                "should_suggest",
                "sum",
            ),

            unique_action_meta=(
                "action_meta",
                "nunique",
            ),
        )
        .reset_index()
    )


    frag[
        "unique_meta_per_suggestion"
    ] = (
        frag[
            "unique_action_meta"
        ]
        /
        frag[
            "suggestions"
        ]
        .clip(
            lower=1
        )
    )


    frag.to_csv(
        out_dir
        / "intent_fragmentation.csv",
        index=False,
    )


    # ========================================================
    # Human annotation template
    # ========================================================

    make_human_annotation(
        df,
        out_dir
        / "human_annotation.csv",
    )


    # ========================================================
    # PNG overview
    # ========================================================

    valid_samples = [
        s
        for s in samples
        if (
            s[
                "sample_id"
            ]
            in rows_by_sample
        )
    ]


    save_overviews(
        valid_samples,
        rows_by_sample,
        out_dir,
    )


    save_summary_plots(
        df,
        out_dir,
        judge_df,
    )


    # ========================================================
    # Summary
    # ========================================================

    summary = make_summary(
        df,
        judge_df,
    )


    summary.to_csv(
        out_dir
        / "summary.csv",
        index=False,
    )


    write_production_readout(
        summary,
        out_dir
        / "PRODUCTION_DECISION.md",
    )


    # ========================================================
    # Console
    # ========================================================

    print(
        "\n"
        "=============================="
    )

    print(
        "SUMMARY"
    )

    print(
        "=============================="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved to:"
    )

    print(
        out_dir.resolve()
    )


# ============================================================
# 28. ENTRY
# ============================================================

if __name__ == "__main__":
    main()
