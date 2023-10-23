from typing import Callable, Union, List, Dict, Iterator


class AsyncScratchpad:
    def __init__(
        self,
        id: str,
        created: float,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop_tokens: Union[str, List[str]],
        function: str,
        stream: bool,
        logger: Callable,
        **unused
    ):
        self.id = id
        self.created = created
        self.finish_reason = ""
        self.temp = min(max(temperature, 0.0), 1.0)
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.function = function
        self.stream = stream
        self._logger = logger
        tmp = stop_tokens
        stop_strings = [tmp] if isinstance(tmp, str) else tmp
        self.metering_generated_tokens_n = 0
        self.metering_total_tokens_n = 0
        self.needs_upload = False
        for k, v in unused.items():
            self.debuglog(f"AsyncScratchpad: unused parameter '{k}' = '{v}'")

    def toplevel_fields(self):
        return {}

    def debuglog(self, *args):
        if self._logger:
            self._logger(*args)

    async def completion(self) -> Iterator[Dict[str, str]]:
        raise NotImplementedError
