import json, os, time
class DeepSeekClient:
    def __init__(self, api_key_env="DEEPSEEK_API_KEY", base_url="https://api.deepseek.com"):
        from openai import OpenAI
        # Fix SSL cert path for conda environments that have broken or missing cacert.pem
        ssl_cert = os.environ.get("SSL_CERT_FILE", "")
        if ssl_cert and not os.path.isfile(ssl_cert):
            try:
                import certifi
                os.environ["SSL_CERT_FILE"] = certifi.where()
            except ImportError:
                os.environ.pop("SSL_CERT_FILE", None)
        api_key=os.environ.get(api_key_env)
        if not api_key: raise RuntimeError(f"Missing API key environment variable: {api_key_env}")
        self.client=OpenAI(api_key=api_key, base_url=base_url)
    def completion(self, max_empty_retries=1, max_connection_retries=5, min_content_chars=0, min_content_retries=3, **kwargs):
        """Call the API, retrying when the response is unusable.

        Two kinds of unusable response are handled:

        * empty content (`finish_reason='length'` with nothing returned) — the historical
          case, retried with a larger `max_tokens`;
        * **truncated but non-empty** content, i.e. shorter than `min_content_chars`. This
          was measured on the env_007 environment analyzer: 2 of 4 calls came back with
          `finish_reason='length'` and 4.6 KB / 9.1 KB instead of the usual 14-20 KB, and
          because the text was non-empty the caller wrote a half-sentence environment card
          to disk and handed it to the reward generator. Callers that know their expected
          output size should set `min_content_chars`; the retry count for this case is
          `min_content_retries` (raising `max_tokens` does not help — the model stops short
          at the same budget, so the useful response is simply a fresh sample).
        """
        from openai import APIConnectionError, APITimeoutError

        base_max_tokens = kwargs.get("max_tokens")
        finish_reason = None
        content = ""
        attempts = max(max_empty_retries, min_content_retries) + 1
        for attempt in range(attempts):
            request_kwargs = dict(kwargs)
            if base_max_tokens and attempt:
                request_kwargs["max_tokens"] = base_max_tokens * (attempt + 1)
            for connection_attempt in range(max_connection_retries + 1):
                try:
                    response = self.client.chat.completions.create(**request_kwargs)
                    break
                except (APIConnectionError, APITimeoutError) as exc:
                    if connection_attempt >= max_connection_retries:
                        raise
                    delay = min(30, 2 ** (connection_attempt + 1))
                    print(
                        f"DeepSeek connection failed ({type(exc).__name__}); "
                        f"retry {connection_attempt + 1}/{max_connection_retries} in {delay}s",
                        flush=True,
                    )
                    time.sleep(delay)
            choice = response.choices[0]
            finish_reason = choice.finish_reason
            message = choice.message
            content = message.content or ""
            if message.tool_calls:
                return response
            if content.strip() and len(content) >= min_content_chars:
                return response
            if attempt < attempts - 1:
                if content.strip():
                    print(
                        f"DeepSeek returned truncated content "
                        f"({len(content)} chars < {min_content_chars}, finish_reason={finish_reason!r}); "
                        f"retry {attempt + 1}/{attempts - 1}",
                        flush=True,
                    )
                else:
                    print(
                        f"DeepSeek returned empty content (finish_reason={finish_reason!r}); "
                        f"retry {attempt + 1}/{attempts - 1} with max_tokens="
                        f"{request_kwargs.get('max_tokens')}",
                        flush=True,
                    )
        raise RuntimeError(
            f"DeepSeek returned an unusable response after {attempts} attempts; "
            f"finish_reason={finish_reason!r}, content_chars={len(content)}, "
            f"required_min_content_chars={min_content_chars}"
        )
    def chat(self, model, system_prompt, user_prompt, temperature=0.2, max_tokens=4096, json_mode=False, min_content_chars=0):
        kwargs={}
        if json_mode: kwargs["response_format"]={"type":"json_object"}
        r=self.completion(model=model,messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],temperature=temperature,max_tokens=max_tokens,min_content_chars=min_content_chars,**kwargs)
        return r.choices[0].message.content
    def chat_json(self,*args,**kwargs): return json.loads(self.chat(*args,json_mode=True,**kwargs))
