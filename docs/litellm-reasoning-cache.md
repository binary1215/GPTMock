# LiteLLM reasoning and cache compatibility

GPTMock keeps the ChatGPT Responses backend behind OpenAI and Ollama adapters. For coding clients through a separate LiteLLM gateway, native Responses is the preferred path because its ordered output items can be replayed without flattening them into one assistant message. The Chat and Ollama extensions below preserve opaque reasoning for clients that retain extra fields. They cannot repair state already discarded by a client or gateway.

This implementation is verified with offline mocked upstream responses. It does not establish real cache-hit rates, provider acceptance of every replay, cost savings, or coding-quality improvement. No gateway configuration or deployed container was changed by these patches.

## Protocol choices

| Client path | GPTMock contract | Remaining condition |
| --- | --- | --- |
| Responses through native LiteLLM Responses | Original reasoning items, encrypted strings, ordered output, native usage and SSE | Client replays original output; gateway must not fall back to Chat |
| Chat to GPTMock Chat | Opt-in `message.reasoning_items` or terminal `delta.reasoning_items`; replay on assistant input | Client and gateway retain the extension, including terminal SSE chunks |
| Ollama to GPTMock chat | Same extension in `message.reasoning_items`; final NDJSON includes usage | Client retains extra fields; an intervening Ollama adapter may discard them |
| Standard Chat content and tool calls only | Text, tools, readable reasoning display | Opaque reasoning continuity cannot be guaranteed |

Readable summaries and opaque reasoning are separate. Neither `<think>` text nor `reasoning_content` is converted into encrypted state. GPTMock does not decode, synthesize, or re-sign provider reasoning. Chat replay preserves reasoning-item order but does not preserve arbitrary interleaving of reasoning, messages, and tools from a Responses output array; use native Responses when that order matters. Keep the default `reasoning_compat=standard` for native fidelity; the opt-in `think-tags` display mode intentionally changes message text. Partially streamed readable summaries are not fully reconciled with terminal summaries; this does not truncate the separately preserved opaque replay items.

## LiteLLM configuration

The following is a starting configuration based on source inspection of LiteLLM 1.103.1, not a deployment verified by this patch. Select a concrete model available to your account. `GPTMOCK_API_BASE` must be reachable from the **LiteLLM container** and end in `/v1`, for example `http://gptmock:8000/v1` on a shared Docker network. Separate hosts require a reachable host address and published GPTMock port. `localhost` inside LiteLLM refers to LiteLLM itself.

```yaml
model_list:
  - model_name: gptmock-responses
    litellm_params:
      model: openai/gpt-5.6-sol
      api_base: os.environ/GPTMOCK_API_BASE
      api_key: os.environ/GPTMOCK_API_KEY
    model_info:
      supported_endpoints:
        - /v1/responses
        - /v1/chat/completions
```

Clients use the gateway's `/v1/responses` endpoint and model alias `gptmock-responses`. The alias name alone does not select a protocol. In the inspected LiteLLM version, `openai/` supports native Responses; `custom_openai/` deployments need the Responses supported-endpoint opt-in to avoid a Chat fallback. Do not force `use_chat_completions_api` on this route. The adapter appends `/responses` to `api_base`; do not configure an endpoint ending in `/v1/responses`. [Provider selection](https://github.com/BerriAI/litellm/blob/580bde9a2d148714889ec1c04a9872819e78a778/litellm/responses/main.py#L486), [URL construction](https://github.com/BerriAI/litellm/blob/580bde9a2d148714889ec1c04a9872819e78a778/litellm/llms/openai/responses/transformation.py#L606)

For Chat clients, an additional gateway conversion remains a separate compatibility boundary. The inspected LiteLLM Responses-to-Chat converter has a single pending reasoning item in its non-stream path, and its stream aggregation helper does not merge `reasoning_items`. Those gateway behaviors are not fixed in this repository. The raw terminal SSE and the gateway's aggregated message must be tested separately. [Chat converter](https://github.com/BerriAI/litellm/blob/580bde9a2d148714889ec1c04a9872819e78a778/litellm/completion_extras/litellm_responses_transformation/transformation.py#L689), [stream aggregation](https://github.com/BerriAI/litellm/blob/580bde9a2d148714889ec1c04a9872819e78a778/litellm/main.py#L8762)

## Native Responses replay

Preserve the complete ordered `output`, not just `output_text` or a reasoning summary. For a caller-owned tool loop, the next `input` consists of previous input, original output items, then matching function results. Reuse the same model/account boundary for opaque state.

```python
history = [{"role": "user", "content": "Inspect the project and suggest a fix."}]
first = client.responses.create(
    model="gptmock-responses", input=history, store=False,
    include=["reasoning.encrypted_content"], prompt_cache_key="project-session-1",
)
history += [item.model_dump(exclude_none=True) for item in first.output]
# Execute only completed function calls using your tool dispatcher.
# history += [{"type": "function_call_output", "call_id": call.call_id, "output": result}]
history += [{"role": "user", "content": "Explain the next step."}]
```

If tools were requested, append their results before a user continuation. GPTMock's non-streaming internal `view_image` loop follows the same order, retaining all output items before appending tool results. A terminal `response.output` is authoritative, even if empty; when absent, completed items are collected by output index with duplicate updates replaced. Native streaming remains caller-owned and does not execute this internal loop.

GPTMock does not add a persistent response store, `GET /responses/{id}`, or automatic `previous_response_id` emulation. Explicit full-history replay is the tested contract. Native usage after an internal `view_image` loop still describes the **final upstream request**, not the aggregate cost of every internal request.

## Chat and Ollama replay

Opaque Chat output is opt-in. Set `GPTMOCK_REASONING_REPLAY=true` in the GPTMock container's `.env`, pass `gptmock serve --reasoning-replay`, or set request `reasoning_replay: true`. A request boolean overrides the server default; `false` suppresses replay output. Explicit incoming assistant `reasoning_items` are honored independently of that output flag. Invalid envelopes or non-boolean request flags return HTTP 400.

```python
history = [{"role": "user", "content": "Plan the implementation."}]
first = client.chat.completions.create(
    model="gptmock-responses", messages=history,
    extra_body={"reasoning_replay": True, "prompt_cache_key": "project-session-1"},
)
# Preserve extension fields, not only role/content/tool_calls.
history.append(first.choices[0].message.model_dump(exclude_none=True))
history.append({"role": "user", "content": "Continue."})
```

This requires a gateway/client that passes GPTMock's extension. Alternatively enable it on GPTMock via environment when the gateway strips unknown top-level request parameters. In streaming Chat, merge `delta.reasoning_items` from the terminal choice chunk before `finish_reason`/`[DONE]` handling discards it. Request `stream_options: {"include_usage": true}` to receive the separate usage chunk; reasoning replay does not depend on that option. Multiple completed reasoning items retain their IDs, opaque strings, and unknown fields without fabricating missing values.

Ollama `/api/chat` accepts and returns assistant `reasoning_items`, separate from readable `thinking`. Streaming puts the items on the final assistant message. `/api/generate` can expose the output extension but has no conversation-message replay input; use `/api/chat` for multi-turn replay. These are GPTMock extensions, not guarantees about standard Ollama clients.

## Cache keys and usage

Cache identity is chosen independently from session routing:

1. A nonempty string `prompt_cache_key` is forwarded unchanged.
2. Otherwise use the `session_id` request header, then body `client_metadata.session_id`.
3. Without either, retain GPTMock's existing in-process instruction/first-user fingerprint mapping to a session UUID.

Blank/null cache keys fall back; non-string cache keys return HTTP 400. The fingerprint mapping is not durable across restarts. GPTMock does not treat a generic metadata user ID as a conversation ID, and does not provide multi-account affinity or cache warming. An explicit stable session/key is preferable across restarts, but a key alone never proves a cache hit.

Chat and legacy completions map native usage as follows. Missing detail objects remain absent; supplied zero values remain zero. Unknown nested detail fields are copied, not stripped.

| Native Responses usage | Chat usage |
| --- | --- |
| `input_tokens` | `prompt_tokens` |
| `input_tokens_details.cached_tokens` | `prompt_tokens_details.cached_tokens` |
| `output_tokens` | `completion_tokens` |
| `output_tokens_details.reasoning_tokens` | `completion_tokens_details.reasoning_tokens` |
| `total_tokens` | `total_tokens` |

Ollama emits standard `prompt_eval_count`/`eval_count` when available plus an additive Chat-shaped `usage` object with the detailed counters. It does not invent local evaluation durations. Cached input remains part of total input; do not subtract it a second time or count reasoning tokens as additional output outside the provider's reported total. Missing cache usage is **unknown**, not evidence of zero hits. This patch makes counters visible; it does not prove that input costs doubled before the patch or that reasoning quality changed after it.

## Regression checks and rollout

```bash
uv sync --dev --frozen
uv run ruff check gptmock/ tests/
uv run pytest tests/ --no-testmon -q
uv build
```

The focused mocked suites are `test_reasoning_replay.py`, `test_responses_continuity.py`, `test_cache_usage_fidelity.py`, and `test_ollama_fidelity.py`. They cover multiple reasoning items, opaque-only items, duplicate done events, terminal-only output, tool continuation, malformed replay, opt-out behavior, cache-key precedence, HTTP session propagation, detailed usage, and streaming/non-streaming adapters. `test_sdk_replay_contracts.py` additionally verifies two-turn Chat and Responses tool/reasoning replay through the installed OpenAI SDK. The offline test guard blocks real HTTPX transports and external socket connections, so a lost mock fails locally. Live tests stay disabled unless explicitly enabled with isolated test credentials.

Before replacing a running service, build this branch's source rather than assuming a published image contains the patch. Verify through the actual LiteLLM gateway that a tool-result second turn retains opaque state, detailed usage reaches the client, and native Responses never falls back to Chat. Measure provider-reported cache counters over repeated stable-prefix requests separately from replay correctness. Failover to a different account/deployment requires an explicit affinity policy; do not silently strip reasoning and call that a successful continuity test.

## Related projects reviewed

The following fixed revisions informed the design on 2026-10-05. This is source-based comparison, not live compatibility certification. No third-party implementation was copied into GPTMock.

| Project and revision | Useful pattern | Limitation to avoid |
| --- | --- | --- |
| [Glour/openai-codex-oauth-proxy](https://github.com/Glour/openai-codex-oauth-proxy/blob/e2aaeaf577d1a8cf4658e394f53a8aa1f3c9df0d/src/server.mjs#L113), MIT | Native input/output preservation; union encrypted include with caller includes | SSE parser is not a comprehensive framing reference |
| [Eunho-J/codex-as-api](https://github.com/Eunho-J/codex-as-api/blob/4e8509e3ef8755a6bd2820e337b2cedf01e2fd45/src/codex_as_api/provider.py#L1219), Apache-2.0 | Account-scoped input plus output replay; explicit cache key first | Chat usage mapping preserves cached tokens but omits reasoning-token detail |
| [EvickaStudio/codex-proxy](https://github.com/EvickaStudio/codex-proxy/blob/88f723881d28879cc46a13a22e1f33cc64df81b2/codex_proxy/upstream.py#L158), Apache-2.0 | Indexed done-item collection with terminal snapshot replacement | Session selection can overwrite explicit cache keys; `setdefault` on include misses encrypted content when a list already exists |
| [dvcrn/codex-oauth-proxy](https://github.com/dvcrn/codex-oauth-proxy/blob/2526b635b9abeab83671b0a6c9bbc0c182c3d7b1/internal/server/transform_responses.go#L73), MIT | Native input retention and fallback cache keys | Include list replacement, basic-only Chat usage, no verified opaque Chat replay |

GPTMock therefore keeps native Responses as its primary fidelity path, adds an explicit Chat replay envelope, preserves detailed usage, and avoids adding an implicit conversation store or account failover behavior in this change.
