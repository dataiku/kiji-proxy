"""Tests for the multilingual dataset pipeline, its backends and the UI (no network)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from model.dataset.label_utils import LabelUtils
from model.dataset.multilingual.backends import (
    ClefDecisionModel,
    ClefLocalDecisionModel,
    DoublewordGenerator,
    FakeDecisionModel,
    FakeGenerator,
    LayaDecisionModel,
    OpenAICompatibleGenerator,
    WorkersAIClient,
    WorkersAIGenerator,
    build_decision_model,
    build_generator,
    describe_backends,
    extract_answers,
)
from model.dataset.multilingual.backends.chat import extract_json, strip_thinking
from model.dataset.multilingual.backends.fake import FAKE_LABELS, FAKE_SAMPLES
from model.dataset.multilingual.backends.laya import state_to_text
from model.dataset.multilingual.clef_server import create_app as create_clef_app
from model.dataset.multilingual.config import (
    GENERATION_LANGUAGES,
    NONE_LABEL,
    TEXT_TYPES,
    PipelineConfig,
)
from model.dataset.multilingual.entity_labeler import (
    MISSING_PII_QUESTION_ID,
    EntityLabeler,
    text_window,
    validate_candidates,
)
from model.dataset.multilingual.labelstudio_tasks import (
    build_labelstudio_task,
    build_regions,
    find_span_offsets,
    write_labelstudio_import,
)
from model.dataset.multilingual.pipeline import GenerationPipeline, build_pipeline
from model.dataset.multilingual.progress import ProgressTracker
from model.dataset.multilingual.prompts import (
    build_generation_prompt,
    get_generation_schema,
)
from model.dataset.multilingual.ui import JobManager, create_app


class RecordingDecisionModel(FakeDecisionModel):
    """Fake decision model that records every request."""

    def __init__(self, labels: dict[str, str] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.labels = labels
        self.requests: list[dict[str, Any]] = []

    def decide(self, state, questions):
        self.requests.append({"state": state, "questions": questions})
        if self.labels is None:
            return super().decide(state, questions)
        candidates = {c["id"]: c["value"] for c in state["candidates"]}
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                answers[qid] = {"probability": 0.1}
                continue
            label = self.labels.get(candidates[int(qid[1:])], NONE_LABEL)
            probs = dict.fromkeys(q["criteria"], 0.0)
            probs[label] = 0.9
            answers[qid] = {"choice": label, "confidence": 0.9, "probabilities": probs}
        return answers


class CannedGenerator:
    """Returns canned texts per language and records prompts."""

    model = "canned"

    def __init__(self, outputs: dict[str, dict[str, Any]]):
        self.outputs = outputs
        self.prompts: list[str] = []

    def generate(self, prompt: str, json_schema: dict[str, Any]) -> dict[str, Any]:
        self.prompts.append(prompt)
        for language, output in self.outputs.items():
            if f"entirely in {language}" in prompt:
                return output
        raise AssertionError("prompt did not name a known language")


JA_TEXT = "田中太郎様、ご注文は〒150-0001 東京都渋谷区神宮前1-2-3へ発送します。連絡先: 090-1234-5678"
KO_TEXT = "김민준 고객님, 주문하신 상품은 서울특별시 강남구 테헤란로 123으로 배송됩니다. 연락처 010-9876-5432"
ZH_TEXT = "王伟先生，您的包裹将寄往北京市朝阳区建国路88号。联系电话 138 0013 8000。"
EN_TEXT = (
    "Hi Anna, Ann Lee from Nolan Logistics will call 087 123 4567 about Anna's parcel."
)


def _labeler(labels: dict[str, str] | None = None, **kwargs) -> EntityLabeler:
    return EntityLabeler(RecordingDecisionModel(labels), **kwargs)


def _mock_transport(handler):
    return httpx.MockTransport(handler)


# ----------------------------------------------------------------- config
def test_ten_languages_including_cjk():
    assert len(GENERATION_LANGUAGES) == 10
    codes = {info["code"] for info in GENERATION_LANGUAGES.values()}
    assert {"ja", "ko", "zh"} <= codes
    assert GENERATION_LANGUAGES["Chinese (Simplified)"]["script"] == "cjk"
    assert len(TEXT_TYPES) >= 10


def test_config_rejects_unknown_language_and_type():
    with pytest.raises(ValueError, match="Unknown languages"):
        PipelineConfig(languages=["Klingon"])
    with pytest.raises(ValueError, match="Unknown text types"):
        PipelineConfig(text_types=["haiku"])
    with pytest.raises(ValueError, match="num_samples"):
        PipelineConfig(num_samples=0)
    with pytest.raises(ValueError, match="max_labels_per_sample"):
        PipelineConfig(min_labels_per_sample=5, max_labels_per_sample=3)
    with pytest.raises(ValueError, match="min_labels_per_sample"):
        PipelineConfig(min_labels_per_sample=0)


# ---------------------------------------------------------------- prompts
def test_generation_prompt_mentions_language_country_type_and_labels():
    labels = {"FIRSTNAME": LabelUtils.LABEL_DESCRIPTIONS["FIRSTNAME"]}
    prompt = build_generation_prompt("Japanese", "Japan", "invoice", labels, 3)
    assert "entirely in Japanese" in prompt
    assert "Japan" in prompt
    assert TEXT_TYPES["invoice"] in prompt
    assert "FIRSTNAME" in prompt
    assert "kanji" in prompt
    assert NONE_LABEL in prompt
    assert get_generation_schema()["required"] == ["text", "candidates"]


def test_generation_prompt_uses_label_descriptions_and_hints():
    labels = {k: LabelUtils.LABEL_DESCRIPTIONS[k] for k in ("USERNAME", "PASSWORD")}
    prompt = build_generation_prompt("Korean", "South Korea", "chat_message", labels)
    assert f"- USERNAME: {LabelUtils.get_label_description('USERNAME')}." in prompt
    assert "- PASSWORD: A password. Use different complexities of passwords." in prompt


# ------------------------------------------------------- JSON extraction
def test_extract_json_strips_thinking_and_fences():
    raw = '<think>let me think</think>\n```json\n{"text": "x", "candidates": []}\n```'
    assert extract_json(raw) == {"text": "x", "candidates": []}
    assert strip_thinking("<think>cut off") == ""
    assert extract_json('Here you go: {"a": [1, 2]} thanks') == {"a": [1, 2]}


# --------------------------------------------------------------- backends
def test_openai_compatible_generator_posts_schema_and_disables_thinking():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        content = '<think>hm</think>{"text": "Hi Anna", "candidates": [{"value": "Anna", "author_label": "FIRSTNAME"}]}'
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    gen = OpenAICompatibleGenerator(
        "Qwen/Qwen3.8-27B",
        "http://vllm/v1/chat/completions",
        api_key="k",
        transport=_mock_transport(handler),
    )
    out = gen.generate("prompt", get_generation_schema())
    assert out["text"] == "Hi Anna"
    assert seen["url"] == "http://vllm/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "Qwen/Qwen3.8-27B"
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["body"]["response_format"]["type"] == "json_schema"


def test_doubleword_generator_uses_api_key_and_priority_tier(monkeypatch):
    monkeypatch.delenv("DOUBLEWORD_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DOUBLEWORD_API_KEY"):
        DoublewordGenerator()
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key")
    gen = DoublewordGenerator("Qwen/Qwen3.8-27B")
    assert gen.api_url == "https://api.doubleword.ai/v1/chat/completions"
    assert gen.headers == {"Authorization": "Bearer dw-key"}
    payload = gen.build_payload("p", get_generation_schema())
    assert payload["service_tier"] == "priority"
    assert "chat_template_kwargs" not in payload


def test_workers_ai_generator_and_clef_use_account_endpoints(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "body": json.loads(request.content)})
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"text": "t", "candidates": []}'}}
                    ]
                },
            )
        return httpx.Response(
            200, json={"success": True, "result": {"c1": {"choice": "EMAIL"}}}
        )

    client = WorkersAIClient(
        base_url="http://cf/client/v4", transport=_mock_transport(handler)
    )
    assert WorkersAIGenerator("@cf/qwen/qwen3.8-27b", client).generate("p", {}) == {
        "text": "t",
        "candidates": [],
    }
    assert ClefDecisionModel("@cf/cloudflare/clef-flash", client).decide(
        {"text": "t"}, {"c1": {"type": "choice"}}
    ) == {"c1": {"choice": "EMAIL"}}
    assert seen[0]["url"] == "http://cf/client/v4/accounts/acct/ai/v1/chat/completions"
    assert (
        seen[1]["url"]
        == "http://cf/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef-flash"
    )
    assert seen[1]["body"] == {
        "state": {"text": "t"},
        "questions": {"c1": {"type": "choice"}},
    }


def test_workers_ai_client_reports_api_errors(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    client = WorkersAIClient(
        transport=_mock_transport(
            lambda r: httpx.Response(
                200, json={"success": False, "errors": [{"message": "nope"}]}
            )
        )
    )
    with pytest.raises(RuntimeError, match="nope"):
        client.run("m", {})


def test_laya_decision_model_sends_text_state_to_systemone():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "answers": {
                    "c1": {
                        "choice": "SURNAME",
                        "probabilities": {"SURNAME": 0.8, "NONE": 0.2},
                    }
                }
            },
        )

    laya = LayaDecisionModel(
        api_url="http://laya:8000/v1/systemone", transport=_mock_transport(handler)
    )
    assert laya.max_state_chars == 1200
    state = {
        "task": "label",
        "text": JA_TEXT,
        "candidates": [{"id": 1, "value": "田中"}],
    }
    answers = laya.decide(
        state, {"c1": {"type": "choice", "instructions": "?", "criteria": {}}}
    )
    assert answers["c1"]["choice"] == "SURNAME"
    assert seen["url"] == "http://laya:8000/v1/systemone"
    assert seen["body"]["state"] == state_to_text(state)
    assert "TEXT:\n" + JA_TEXT in seen["body"]["state"]
    assert "CANDIDATES:\n1. 田中" in seen["body"]["state"]
    assert "model" not in seen["body"]


def test_clef_local_decision_model_posts_structured_state_to_systemone(monkeypatch):
    monkeypatch.delenv("CLEF_API_URL", raising=False)
    monkeypatch.setenv("CLEF_API_KEY", "local-key")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "Cloudflare/clef-flash",
                "answers": {
                    "c1": {
                        "type": "choice",
                        "choice": "SURNAME",
                        "confidence": 0.8,
                        "probabilities": {"SURNAME": 0.8, "NONE": 0.2},
                    },
                    "missing_pii": {"type": "noul", "noul": 0.25},
                },
            },
        )

    clef = build_decision_model("clef-local")
    assert isinstance(clef, ClefLocalDecisionModel)
    assert clef.max_state_chars is None
    clef._http = httpx.Client(transport=_mock_transport(handler))
    state = {"text": JA_TEXT, "candidates": [{"id": 1, "value": "田中"}]}
    questions = {
        "c1": {"type": "choice", "instructions": "?", "criteria": {"SURNAME": "s"}},
        "missing_pii": {"type": "noul", "instructions": "?"},
    }
    answers = clef.decide(state, questions)
    assert seen["url"] == "http://localhost:8002/v1/systemone"
    assert seen["auth"] == "Bearer local-key"
    assert seen["body"] == {
        "model": "Cloudflare/clef-flash",
        "state": state,
        "questions": questions,
    }
    assert answers["c1"]["choice"] == "SURNAME"
    assert EntityLabeler.parse_noul(answers["missing_pii"]) == 0.25


def test_clef_server_serves_systemone_and_checks_api_key():
    seen = []

    def answer(request: dict[str, Any]) -> dict[str, Any]:
        seen.append(request)
        if not request.get("questions"):
            raise ValueError("at least one question is required")
        return {"model": request["model"], "answers": {"q": {"noul": 0.5}}}

    client = TestClient(create_clef_app(answer, "Cloudflare/clef-flash", "key"))
    assert client.get("/health").json()["model"] == "Cloudflare/clef-flash"
    body = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}}
    assert client.post("/v1/systemone", json=body).status_code == 401
    auth = {"Authorization": "Bearer key"}
    response = client.post("/v1/systemone", json=body, headers=auth)
    assert response.json()["answers"] == {"q": {"noul": 0.5}}
    assert seen[0]["model"] == "Cloudflare/clef-flash"  # filled in when omitted
    assert (
        client.post("/v1/systemone", json={"state": "s"}, headers=auth).status_code
        == 400
    )


def test_extract_answers_handles_nested_containers():
    nested = {"answers": {"c1": {"choice": "EMAIL"}, "other": 1}}
    assert extract_answers(nested, {"c1"}) == {"c1": {"choice": "EMAIL"}}
    assert extract_answers({"c1": {"choice": "EMAIL"}, "c2": "SURNAME"}, {"c1", "c2"})[
        "c2"
    ] == {"value": "SURNAME"}
    with pytest.raises(ValueError):
        extract_answers({"unrelated": {}}, {"c1"})


def test_registry_builds_backends_and_rejects_unknown_names():
    assert isinstance(build_generator("fake"), FakeGenerator)
    assert isinstance(build_decision_model("fake"), FakeDecisionModel)
    laya = build_decision_model("laya", api_url="http://x/v1/systemone")
    assert (
        isinstance(laya, LayaDecisionModel) and laya.api_url == "http://x/v1/systemone"
    )
    local = build_generator("openai-compatible", model="my-qwen")
    assert isinstance(local, OpenAICompatibleGenerator) and local.model == "my-qwen"
    with pytest.raises(ValueError, match="Unknown generator backend"):
        build_generator("nope")
    described = describe_backends()
    assert [b["name"] for b in described["generators"]] == [
        "workers-ai",
        "openai-compatible",
        "doubleword",
        "fake",
    ]
    assert [b["name"] for b in described["decision_models"]] == [
        "clef",
        "clef-local",
        "laya",
        "fake",
    ]


def test_fake_backends_cover_every_language():
    assert set(FAKE_SAMPLES) == set(GENERATION_LANGUAGES)
    for text, pairs in FAKE_SAMPLES.values():
        for value, _ in pairs:
            assert value in text
    gen = FakeGenerator()
    out = gen.generate(
        build_generation_prompt("Korean", "South Korea", "invoice", {}), {}
    )
    assert out["text"].startswith("[invoice] ")
    assert FAKE_LABELS["김"] == "SURNAME"


# ---------------------------------------------------------------- labeler
def test_validate_candidates_keeps_verbatim_cjk_spans_and_drops_others():
    kept, dropped = validate_candidates(
        JA_TEXT,
        [
            {"value": "田中", "author_label": "SURNAME"},
            {"value": "田中", "author_label": "SURNAME"},
            {"value": "山田", "author_label": "SURNAME"},
            {"value": " ", "author_label": NONE_LABEL},
        ],
    )
    assert [c["value"] for c in kept] == ["田中"]
    assert dropped == ["山田"]


def test_labeler_builds_choice_questions_with_none_option_and_noul():
    labeler = _labeler({"田中": "SURNAME", "太郎": "FIRSTNAME"})
    result = labeler.label(
        JA_TEXT,
        "Japanese",
        "Japan",
        [
            {"value": "田中", "author_label": "SURNAME"},
            {"value": "太郎", "author_label": "SURNAME"},
            {"value": "渋谷区", "author_label": "CITY"},
        ],
    )
    requests = labeler.decision_model.requests
    assert len(requests) == 1
    questions = requests[0]["questions"]
    assert set(questions) == {"c1", "c2", "c3", MISSING_PII_QUESTION_ID}
    assert questions["c1"]["type"] == "choice"
    assert set(questions["c1"]["criteria"]) == set(LabelUtils.STANDARD_PII_LABELS) | {
        NONE_LABEL
    }
    assert questions["c1"]["criteria"][
        "CREDITCARDNUMBER"
    ] == LabelUtils.get_label_description("CREDITCARDNUMBER")
    assert questions[MISSING_PII_QUESTION_ID]["type"] == "noul"
    assert requests[0]["state"]["text"] == JA_TEXT

    labels = {span.value: span.label for span in result.spans}
    assert labels == {"田中": "SURNAME", "太郎": "FIRSTNAME", "渋谷区": NONE_LABEL}
    assert [s.value for s in result.pii_spans] == ["田中", "太郎"]
    assert result.rejected_spans[0].value == "渋谷区"
    assert result.spans[1].author_label == "SURNAME"  # the author disagreed
    assert result.spans[0].score == pytest.approx(0.9)
    assert result.missing_pii_probability == pytest.approx(0.1)


def test_labeler_chunks_requests_to_backend_limit():
    text = " ".join(f"tok{i}" for i in range(70))
    candidates = [{"value": f"tok{i}", "author_label": NONE_LABEL} for i in range(70)]
    labeler = _labeler({}, max_questions_per_request=64)
    result = labeler.label(text, "English", "United States", candidates)
    assert len(result.spans) == 70
    sizes = [len(r["questions"]) for r in labeler.decision_model.requests]
    assert sizes == [64, 7]  # 63 candidates + noul, then the remaining 7
    assert MISSING_PII_QUESTION_ID in labeler.decision_model.requests[0]["questions"]
    assert (
        MISSING_PII_QUESTION_ID not in labeler.decision_model.requests[1]["questions"]
    )


def test_labeler_uses_text_windows_for_small_context_models():
    model = RecordingDecisionModel({"Anna": "FIRSTNAME", "Lee": "SURNAME"})
    model.max_state_chars = 40
    labeler = EntityLabeler(model)
    result = labeler.label(
        EN_TEXT,
        "English",
        "Ireland",
        [
            {"value": "Anna", "author_label": "FIRSTNAME"},
            {"value": "Lee", "author_label": "SURNAME"},
        ],
    )
    assert [(s.value, s.label) for s in result.spans] == [
        ("Anna", "FIRSTNAME"),
        ("Lee", "SURNAME"),
    ]
    assert result.missing_pii_probability is None
    assert len(model.requests) == 2
    for request in model.requests:
        assert len(request["state"]["text"]) <= 40
        assert set(request["questions"]) == {"c1"}
    assert "Lee" in model.requests[1]["state"]["text"]
    assert (
        text_window("a" * 100 + "needle" + "b" * 100, "needle", 20).count("needle") == 1
    )
    assert len(text_window("a" * 100 + "needle" + "b" * 100, "needle", 20)) == 20


def test_parse_choice_falls_back_to_argmax_and_normalises_case():
    labeler = _labeler()
    label, score, probs = labeler.parse_choice(
        {"probabilities": {"email": 0.7, "none": 0.3}}
    )
    assert (label, score) == ("EMAIL", pytest.approx(0.7))
    assert probs == {"EMAIL": 0.7, NONE_LABEL: 0.3}
    label, score, _ = labeler.parse_choice(
        {
            "choice": "SURNAME",
            "confidence": 0.55,
            "probabilities": [{"option": "SURNAME", "probability": 0.55}],
        }
    )
    assert (label, score) == ("SURNAME", pytest.approx(0.55))
    assert EntityLabeler.parse_noul({"probabilities": {"true": 0.2}}) == 0.2


# ----------------------------------------------------------- Label Studio
def test_find_span_offsets_respects_word_boundaries_for_latin_only():
    assert find_span_offsets(EN_TEXT, "Ann") == [(9, 12)]
    second = EN_TEXT.index("Anna's")
    assert find_span_offsets(EN_TEXT, "Anna") == [(3, 7), (second, second + 4)]
    assert find_span_offsets(ZH_TEXT, "王伟", script="cjk") == [(0, 2)]
    assert find_span_offsets(KO_TEXT, "010-9876-5432", script="cjk") == [
        (KO_TEXT.index("010"), len(KO_TEXT))
    ]


def test_build_regions_prefers_longer_spans_and_avoids_overlap():
    mask = [
        {"value": "Lee", "label": "SURNAME", "score": 0.8},
        {"value": "Ann Lee", "label": "FIRSTNAME", "score": 0.6},
        {"value": "Nolan Logistics", "label": "COMPANYNAME", "score": 0.95},
        {"value": "Hi", "label": NONE_LABEL, "score": 0.9},
    ]
    regions = build_regions(EN_TEXT, mask)
    assert [(r["value"]["text"], r["value"]["labels"][0]) for r in regions] == [
        ("Ann Lee", "FIRSTNAME"),
        ("Nolan Logistics", "COMPANYNAME"),
    ]
    assert [r["id"] for r in regions] == ["ent-1", "ent-2"]
    assert regions[1]["score"] == pytest.approx(0.95)


def test_build_labelstudio_task_has_offsets_scores_and_metadata():
    sample = {
        "text": ZH_TEXT,
        "language": "Chinese (Simplified)",
        "language_code": "zh",
        "script": "cjk",
        "country": "China",
        "text_type": "chat_message",
        "generator_model": "@cf/qwen/qwen3.8-27b",
        "labeler_model": "@cf/cloudflare/clef-flash",
        "sample_index": 7,
        "missing_pii_probability": 0.05,
        "privacy_mask": [
            {
                "value": "王",
                "label": "SURNAME",
                "score": 0.9,
                "author_label": "SURNAME",
            },
            {
                "value": "伟",
                "label": "FIRSTNAME",
                "score": 0.8,
                "author_label": "SURNAME",
            },
            {"value": "北京市", "label": "CITY", "score": 0.3, "author_label": "CITY"},
            {
                "value": "138 0013 8000",
                "label": "PHONENUMBER",
                "score": 0.99,
                "author_label": "PHONENUMBER",
            },
        ],
        "rejected_candidates": [{"value": "包裹", "label": NONE_LABEL, "score": 0.9}],
    }
    task = build_labelstudio_task(sample, min_score=0.5)
    assert task["data"]["text"] == ZH_TEXT
    assert task["data"]["language_code"] == "zh"
    prediction = task["predictions"][0]
    assert (
        prediction["model_version"] == "@cf/qwen/qwen3.8-27b+@cf/cloudflare/clef-flash"
    )
    result = prediction["result"]
    assert [r["value"]["labels"][0] for r in result] == [
        "SURNAME",
        "FIRSTNAME",
        "PHONENUMBER",
    ]
    for region in result:
        start, end = region["value"]["start"], region["value"]["end"]
        assert ZH_TEXT[start:end] == region["value"]["text"]
    assert prediction["score"] == pytest.approx((0.9 + 0.8 + 0.99) / 3, abs=1e-3)
    assert task["meta"]["author_agreement"] == pytest.approx(0.75)
    assert task["meta"]["rejected_candidates"] == ["包裹"]


def test_write_labelstudio_import_is_json_array(tmp_path):
    path = write_labelstudio_import([{"data": {"text": "한국어"}}], tmp_path / "x.json")
    assert json.loads(path.read_text(encoding="utf-8")) == [
        {"data": {"text": "한국어"}}
    ]
    assert "한국어" in path.read_text(encoding="utf-8")


def test_every_configured_label_has_description_and_unique_color():
    for code, info in LabelUtils.LABEL_DESCRIPTIONS.items():
        assert info.get("description"), f"{code} has no description"
    colors = [info["color"] for info in LabelUtils.LABEL_DESCRIPTIONS.values()]
    assert len(colors) == len(set(colors)), "two labels share a color"


def test_generated_labelstudio_config_contains_every_label(monkeypatch):
    xml = LabelUtils.generate_labelstudio_config()
    for code in LabelUtils.STANDARD_PII_LABELS:
        assert f'<Label value="{code}"' in xml, f"{code} missing from config"
    patched = dict(LabelUtils.LABEL_DESCRIPTIONS)
    patched["MEDICALRECORDNUM"] = {
        "name": "Medical record number",
        "description": "x",
        "color": "#00acc1",
        "chance": 0.1,
        "hints": [],
    }
    monkeypatch.setattr(LabelUtils, "LABEL_DESCRIPTIONS", patched)
    assert (
        '<Label value="MEDICALRECORDNUM" background="#00acc1"/>'
        in LabelUtils.generate_labelstudio_config()
    )
    assert 'granularity="symbol"' in LabelUtils.generate_labelstudio_config(
        granularity="symbol"
    )


def test_committed_labeling_config_is_up_to_date():
    committed = Path("model/dataset/labelstudio/LabelingConfig.txt").read_text()
    assert committed == LabelUtils.generate_labelstudio_config()


# --------------------------------------------------------------- progress
def test_progress_tracker_counts_and_snapshot():
    updates = []
    tracker = ProgressTracker(on_update=updates.append)
    tracker.start(3, "/out")
    tracker.record("created", "Japanese", "invoice")
    tracker.record("failed", "Korean", "invoice", error="boom", index=1)
    snap = tracker.snapshot()
    assert (snap["status"], snap["done"], snap["created"], snap["failed"]) == (
        "running",
        2,
        1,
        1,
    )
    assert snap["percent"] == pytest.approx(66.7)
    assert snap["per_language"]["Korean"] == {"failed": 1}
    assert snap["errors"][0]["error"] == "boom"
    tracker.request_cancel()
    assert tracker.cancelled and tracker.snapshot()["status"] == "cancelling"
    tracker.finish({"samples": 1}, "/out/tasks.json")
    assert tracker.snapshot()["status"] == "cancelled"
    assert len(updates) >= 5
    with pytest.raises(ValueError):
        tracker.record("weird", "x", "y")


# --------------------------------------------------------------- pipeline
def test_plan_balances_languages_and_rotates_text_types():
    config = PipelineConfig(
        num_samples=30,
        languages=["English", "Japanese", "Korean"],
        text_types=["invoice", "chat_message"],
        seed=3,
    )
    pipeline = GenerationPipeline(config, generator=None, labeler=None)
    plans = [pipeline.plan(i) for i in range(12)]
    assert [p.language for p in plans[:4]] == [
        "English",
        "Japanese",
        "Korean",
        "English",
    ]
    assert [p.text_type for p in plans[:7]] == ["invoice"] * 3 + [
        "chat_message"
    ] * 3 + ["invoice"]
    assert plans[1].country == "Japan" and plans[1].language_code == "ja"
    assert plans[0].file_name == "00000_en_invoice.json"
    assert pipeline.plan(5) == plans[5]


def test_plan_draws_a_random_label_count_per_sample():
    config = PipelineConfig(
        num_samples=40,
        languages=["English"],
        text_types=["invoice"],
        min_labels_per_sample=2,
        max_labels_per_sample=6,
        seed=1,
    )
    pipeline = GenerationPipeline(config, generator=None, labeler=None)
    counts = [len(pipeline.plan(i).labels) for i in range(40)]
    # Address labels are expanded as a group, so counts may exceed the draw,
    # but never fall below the minimum, and the draw itself varies.
    assert min(counts) >= 2
    assert len(set(counts)) > 1
    assert [len(pipeline.plan(i).labels) for i in range(40)] == counts  # seeded

    fixed = PipelineConfig(
        num_samples=5,
        languages=["English"],
        text_types=["invoice"],
        min_labels_per_sample=3,
        max_labels_per_sample=3,
    )
    fixed_pipeline = GenerationPipeline(fixed, generator=None, labeler=None)
    assert all(len(fixed_pipeline.plan(i).labels) >= 3 for i in range(5))

    other_seed = PipelineConfig(
        num_samples=40, languages=["English"], text_types=["invoice"], seed=2
    )
    assert [
        len(GenerationPipeline(other_seed, None, None).plan(i).labels)
        for i in range(40)
    ] != counts


def test_pipeline_end_to_end_writes_resumable_samples_and_import(tmp_path):
    config = PipelineConfig(
        num_samples=4,
        output_dir=str(tmp_path),
        languages=["Japanese", "Korean", "Chinese (Simplified)", "English"],
        text_types=["business_email"],
        max_workers=2,
        min_score=0.5,
        generator_backend="fake",
        decision_backend="fake",
    )
    generator = CannedGenerator(
        {
            "Japanese": {
                "text": JA_TEXT,
                "candidates": [
                    {"value": "田中", "author_label": "SURNAME"},
                    {"value": "太郎", "author_label": "FIRSTNAME"},
                    {"value": "090-1234-5678", "author_label": "PHONENUMBER"},
                    {"value": "佐藤", "author_label": "SURNAME"},
                ],
            },
            "Korean": {
                "text": KO_TEXT,
                "candidates": [
                    {"value": "김", "author_label": "SURNAME"},
                    {"value": "민준", "author_label": "FIRSTNAME"},
                    {"value": "테헤란로", "author_label": "STREET"},
                ],
            },
            "Chinese (Simplified)": {
                "text": ZH_TEXT,
                "candidates": [
                    {"value": "王伟", "author_label": "FIRSTNAME"},
                    {"value": "包裹", "author_label": NONE_LABEL},
                ],
            },
            "English": {
                "text": EN_TEXT,
                "candidates": [
                    {"value": "Anna", "author_label": "FIRSTNAME"},
                    {"value": "Nolan Logistics", "author_label": "COMPANYNAME"},
                ],
            },
        }
    )
    labeler = _labeler(
        {
            "田中": "SURNAME",
            "太郎": "FIRSTNAME",
            "090-1234-5678": "PHONENUMBER",
            "김": "SURNAME",
            "민준": "FIRSTNAME",
            "테헤란로": "STREET",
            "王伟": "SURNAME",
            "Anna": "FIRSTNAME",
            "Nolan Logistics": "COMPANYNAME",
        }
    )
    pipeline = GenerationPipeline(config, generator, labeler)
    tracker = ProgressTracker()
    tasks = pipeline.run(tracker, show_progress_bar=False)

    names = sorted(p.name for p in (tmp_path / "samples").glob("*.json"))
    assert names == [
        "00000_ja_business_email.json",
        "00001_ko_business_email.json",
        "00002_zh_business_email.json",
        "00003_en_business_email.json",
    ]
    ja = json.loads((tmp_path / "samples" / names[0]).read_text(encoding="utf-8"))
    assert ja["dropped_candidates"] == ["佐藤"]
    assert ja["generator_backend"] == "fake" and ja["labeler_backend"] == "fake"
    assert {m["value"]: m["label"] for m in ja["privacy_mask"]} == {
        "田中": "SURNAME",
        "太郎": "FIRSTNAME",
        "090-1234-5678": "PHONENUMBER",
    }
    zh = json.loads((tmp_path / "samples" / names[2]).read_text(encoding="utf-8"))
    assert (
        zh["privacy_mask"][0]["author_label"] == "FIRSTNAME"
        and zh["privacy_mask"][0]["label"] == "SURNAME"
    )

    assert len(tasks) == 4
    assert (
        json.loads((tmp_path / "labelstudio_tasks.json").read_text(encoding="utf-8"))
        == tasks
    )
    ko_task = next(t for t in tasks if t["data"]["language_code"] == "ko")
    assert [r["value"]["labels"][0] for r in ko_task["predictions"][0]["result"]] == [
        "SURNAME",
        "FIRSTNAME",
        "STREET",
    ]
    assert 'granularity="symbol"' in (tmp_path / "labeling_config.xml").read_text()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["samples"] == 4 and summary["labels"]["SURNAME"] == 3
    assert 0 < summary["author_labeler_agreement"] < 1
    snap = tracker.snapshot()
    assert (
        snap["status"] == "completed"
        and snap["created"] == 4
        and snap["summary"] == summary
    )
    assert json.loads((tmp_path / "progress.json").read_text())["status"] == "completed"

    # Resume: nothing regenerated, export rebuilt from disk, outcomes are "skipped".
    before = len(generator.prompts)
    second = ProgressTracker()
    pipeline.run(second, show_progress_bar=False)
    assert len(generator.prompts) == before
    assert second.snapshot()["skipped"] == 4


def test_pipeline_keeps_going_when_one_sample_fails(tmp_path):
    class Flaky(CannedGenerator):
        def generate(self, prompt, json_schema):
            if "entirely in Korean" in prompt:
                raise RuntimeError("boom")
            return super().generate(prompt, json_schema)

    config = PipelineConfig(
        num_samples=2,
        output_dir=str(tmp_path),
        languages=["English", "Korean"],
        text_types=["invoice"],
        max_workers=1,
    )
    generator = Flaky(
        {
            "English": {
                "text": EN_TEXT,
                "candidates": [{"value": "Anna", "author_label": "FIRSTNAME"}],
            }
        }
    )
    tracker = ProgressTracker()
    tasks = GenerationPipeline(config, generator, _labeler({"Anna": "FIRSTNAME"})).run(
        tracker, show_progress_bar=False
    )
    assert len(tasks) == 1 and tasks[0]["data"]["language_code"] == "en"
    snap = tracker.snapshot()
    assert (
        snap["failed"] == 1
        and snap["errors"][0]["error"] == "boom"
        and snap["status"] == "completed"
    )


def test_pipeline_cancel_skips_unstarted_samples(tmp_path):
    config = PipelineConfig(
        num_samples=6,
        output_dir=str(tmp_path),
        languages=["English"],
        text_types=["invoice"],
        max_workers=1,
        generator_backend="fake",
        decision_backend="fake",
    )
    tracker = ProgressTracker()

    class CancellingGenerator(FakeGenerator):
        def generate(self, prompt, json_schema):
            tracker.request_cancel()
            return super().generate(prompt, json_schema)

    pipeline = GenerationPipeline(
        config, CancellingGenerator(), EntityLabeler(FakeDecisionModel())
    )
    tasks = pipeline.run(tracker, show_progress_bar=False)
    snap = tracker.snapshot()
    assert snap["status"] == "cancelled"
    assert snap["created"] == 1 and snap["cancelled"] == 5
    assert len(tasks) == 1


def test_build_pipeline_wires_fake_backends(tmp_path):
    config = PipelineConfig(
        num_samples=1,
        output_dir=str(tmp_path),
        generator_backend="fake",
        decision_backend="fake",
    )
    pipeline = build_pipeline(config)
    assert isinstance(pipeline.generator, FakeGenerator)
    assert isinstance(pipeline.labeler.decision_model, FakeDecisionModel)


# --------------------------------------------------------------------- UI
def _wait(client: TestClient, job_id: str, timeout: float = 15.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["progress"]["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_ui_options_and_page():
    client = TestClient(create_app(JobManager()))
    assert "Multilingual PII dataset generator" in client.get("/").text
    options = client.get("/api/options").json()
    assert len(options["languages"]) == 10
    assert options["backends"]["generators"][0]["name"] == "workers-ai"
    assert options["defaults"]["decision_backend"] == "clef"


def test_ui_starts_tracks_and_exports_a_job(tmp_path):
    def factory(config: PipelineConfig) -> GenerationPipeline:
        return GenerationPipeline(
            config, FakeGenerator(), EntityLabeler(FakeDecisionModel())
        )

    client = TestClient(create_app(JobManager(pipeline_factory=factory)))
    bad = client.post(
        "/api/jobs",
        json={"num_samples": 2, "languages": ["Klingon"], "output_dir": str(tmp_path)},
    )
    assert bad.status_code == 400 and "Unknown languages" in bad.json()["detail"]

    res = client.post(
        "/api/jobs",
        json={
            "num_samples": 3,
            "languages": ["Japanese", "Korean", "Chinese (Simplified)"],
            "text_types": ["support_ticket"],
            "generator_backend": "fake",
            "decision_backend": "fake",
            "output_dir": str(tmp_path / "run"),
        },
    )
    assert res.status_code == 201
    job_id = res.json()["id"]
    job = _wait(client, job_id)
    assert job["progress"]["created"] == 3 and job["progress"]["percent"] == 100.0
    assert job["progress"]["summary"]["languages"] == {
        "Chinese (Simplified)": 1,
        "Japanese": 1,
        "Korean": 1,
    }
    assert [j["id"] for j in client.get("/api/jobs").json()] == [job_id]

    samples = client.get(f"/api/jobs/{job_id}/samples?limit=2").json()
    assert len(samples) == 2 and samples[0]["predictions"][0]["result"]
    export = client.get(f"/api/jobs/{job_id}/tasks")
    assert export.status_code == 200 and len(export.json()) == 3
    assert (tmp_path / "run" / "progress.json").exists()
    assert client.get("/api/jobs/nope").status_code == 404


def test_ui_credential_errors_are_reported_as_400(monkeypatch, tmp_path):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    client = TestClient(create_app(JobManager()))
    res = client.post("/api/jobs", json={"num_samples": 1, "output_dir": str(tmp_path)})
    assert res.status_code == 400 and "CLOUDFLARE_ACCOUNT_ID" in res.json()["detail"]


def test_ui_cancel_stops_a_running_job(tmp_path):
    def factory(config: PipelineConfig) -> GenerationPipeline:
        return GenerationPipeline(
            config, FakeGenerator(delay=0.2), EntityLabeler(FakeDecisionModel())
        )

    client = TestClient(create_app(JobManager(pipeline_factory=factory)))
    res = client.post(
        "/api/jobs",
        json={
            "num_samples": 20,
            "languages": ["English"],
            "generator_backend": "fake",
            "decision_backend": "fake",
            "max_workers": 1,
            "output_dir": str(tmp_path),
        },
    )
    job_id = res.json()["id"]
    time.sleep(0.3)
    assert client.post(f"/api/jobs/{job_id}/cancel").json()["progress"]["status"] in (
        "cancelling",
        "cancelled",
    )
    job = _wait(client, job_id)
    assert job["progress"]["status"] == "cancelled"
    assert job["progress"]["cancelled"] > 0 and job["progress"]["created"] < 20
