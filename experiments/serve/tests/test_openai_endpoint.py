"""Local (no-device, no-fastapi) unit test for the OpenAI endpoint translation.

Covers the pure helpers (OpenAI schema + chat-template wiring). The Unix-socket
bridge + the live HTTP path are covered by the e2e (server up + curl) documented
in the endpoint module. Run from the repo root:

    python -m experiments.serve.tests.test_openai_endpoint
"""
import os

os.environ.setdefault("TT_OPENAI_BUILD_APP", "0")  # don't import fastapi/transformers

from experiments.serve import openai_endpoint as oa  # noqa: E402


class _FakeTok:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "".join(f"<|{m['role']}|>{m['content']}" for m in messages) + "<|assistant|>"


def test_messages_to_prompt():
    p = oa._messages_to_prompt(_FakeTok(), [{"role": "user", "content": "hi"}])
    assert p == "<|user|>hi<|assistant|>", p


def test_chat_completion_schema():
    r = oa._chat_completion("Paris.", "m", 5, 2, "stop")
    assert r["object"] == "chat.completion"
    assert r["choices"][0]["message"] == {"role": "assistant", "content": "Paris."}
    assert r["choices"][0]["finish_reason"] == "stop"
    assert r["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}


def test_chat_chunk_schema():
    assert oa._chat_chunk("Par", "m", "id1")["choices"][0]["delta"] == {"content": "Par"}
    assert oa._chat_chunk("", "m", "id1", role=True)["choices"][0]["delta"] == {"role": "assistant"}
    assert oa._chat_chunk("", "m", "id1", finish="stop")["choices"][0]["finish_reason"] == "stop"
    assert oa._chat_chunk("Par", "m", "id1")["object"] == "chat.completion.chunk"


if __name__ == "__main__":
    test_messages_to_prompt()
    test_chat_completion_schema()
    test_chat_chunk_schema()
    print("test_openai_endpoint: 3/3 PASS")
