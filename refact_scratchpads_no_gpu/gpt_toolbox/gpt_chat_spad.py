import asyncio
import functools
import json
from typing import List, Tuple, Dict, Union, Iterator

from refact_scratchpads_no_gpu.async_scratchpad import ascratch

import openai
import tiktoken


def gpt_prices(  # Apr 4 2023:
    model_name: str,
) -> Tuple[int, int]:
    # GPT-4 8K prompt[$0.03 / 1K tokens] generated[$0.06 / 1K tokens]
    if model_name.startswith("gpt-4") or model_name.startswith("gpt4"):
        pp1000t_prompt = 30_000
        pp1000t_generated = 60_000
    # gpt-3.5-turbo $0.002 / 1K tokens
    elif model_name.startswith("gpt-3.5-turbo"):
        pp1000t_prompt = 2_000
        pp1000t_generated = 2_000
    else:
        raise ValueError(f'get_prices: Unknown model: {model_name}')
    return pp1000t_prompt, pp1000t_generated


@functools.lru_cache(maxsize=10)
def engine_to_encoding(engine: str) -> tiktoken.Encoding:
    return tiktoken.encoding_for_model(engine)


ACCUMULATE_N_STREAMING_CHUNKS = 5
engine_to_encoding("text-davinci-003")  # this immediately tests if tiktoken works or not


def calculate_chat_tokens(model_name, messages, completion):
    enc = engine_to_encoding(model_name)
    calc_prompt_tokens_n = 2   # warmup
    for d in messages:
        calc_prompt_tokens_n += len(enc.encode(d["content"], disallowed_special=()))
        calc_prompt_tokens_n += len(enc.encode(d["role"], disallowed_special=()))
        calc_prompt_tokens_n += 4    # to switch user/assistant
    calc_generated_tokens_n = len(enc.encode(completion, disallowed_special=())) + 2   # one to switch, another EOF
    return calc_prompt_tokens_n, calc_generated_tokens_n


class GptChat(ascratch.AsyncScratchpad):
    def __init__(
            self,
            id: str,
            *,
            created: float,
            temperature: float,
            top_p: float,
            max_tokens: int,
            stop_tokens: Union[str, List[str]],
            messages: List[Dict[str, str]],
            model: str,  # always "longthink", don't use
            **more,
    ):
        super().__init__(
            id=id,
            created=created,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_tokens=stop_tokens,
            **more,
        )

        self._model_name = "gpt-3.5-turbo"
        if "gpt4" in self.function or "gpt-4" in self.function:
            self._model_name = "gpt-4"
        self._stream_timeout_sec = 15

        messages = messages or []
        if not messages or messages[0].get('role') != 'system':
            messages = [
                {
                    "role": "system",
                    "content": "You are a coding assistant that outputs short answers, give links to documentation.",
                }, *messages
            ]
        self._messages = messages
        self._completion = ""

    @property
    def prices(self) -> Tuple[int, int]:
        return gpt_prices(self._model_name)

    async def completion(self) -> Iterator[Dict[str, str]]:
        gen = await openai.ChatCompletion.acreate(
            model=self._model_name,
            messages=self._messages,
            max_tokens=self.max_tokens,
            temperature=self.temp,
            stream=True,
        )
        accum = ""
        role = ""
        tokens = 0
        self.metering_prompt_tokens_n = 0
        self.metering_generated_tokens_n = 0
        try:
            def forward_streaming():
                nonlocal tokens, accum, role
                self._completion += accum
                msg = {
                    "chat__role": "assistant",
                    "chat__content": self._completion,
                }
                accum = ""
                return msg

            while True:
                resp = await asyncio.wait_for(gen.__anext__(), self._stream_timeout_sec)
                delta = resp.choices[0].delta
                if "role" in delta:
                    role = delta["role"]
                if "content" in delta:
                    accum += delta["content"]
                    tokens += 1  # assuming 1 token per chunk
                if "swear" in accum:
                    raise ValueError("swear!")
                if "finish_reason" in resp.choices[0] and resp.choices[0]["finish_reason"] is not None:
                    self.finish_reason = resp.choices[0]["finish_reason"]
                if self.finish_reason:
                    break
                if tokens % ACCUMULATE_N_STREAMING_CHUNKS == 0:
                    yield forward_streaming()
                if self.finish_reason:  # cancelled from main coroutine
                    break
            if self.finish_reason == "":
                self.finish_reason = "END"
        except asyncio.exceptions.TimeoutError as e:
            self.debuglog("CHAT TIMEOUT:", str(type(e)), str(e))
        except Exception as e:
            self.debuglog("CHAT EXCEPTION:", str(type(e)), str(e))
            self.finish_reason = "ERROR"
        yield forward_streaming()

    def toplevel_fields(self):
        if not self.finish_reason:
            return {}
        calc_prompt_tokens_n, calc_generated_tokens_n = calculate_chat_tokens(
            self._model_name, self._messages, self._completion
        )
        self.metering_prompt_tokens_n = calc_prompt_tokens_n
        self.metering_generated_tokens_n = calc_generated_tokens_n
        metering_message = {
            "metering_prompt_tokens_n": self.metering_prompt_tokens_n,
            "metering_generated_tokens_n": self.metering_generated_tokens_n,
            "pp1000t_prompt": self.prices[0],
            "pp1000t_generated": self.prices[1],
            "model_name": self._model_name,
        }
        self.debuglog(json.dumps(metering_message))
        return metering_message
