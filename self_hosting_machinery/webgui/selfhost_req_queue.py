import asyncio
import random

from self_hosting_machinery.webgui import selfhost_webutils

from typing import Dict, Any


__all__ = ["Ticket"]


# TODO: why not uuid???
def random_guid(n=12):
    random_chars = "0123456789" + "ABCDEFGHIJKLNMPQRSTUVWXYZ" + "ABCDEFGHIJKLNMPQRSTUVWXYZ".lower()
    return "".join(
        [
            random_chars[random.randint(0, len(random_chars) - 1)]
            for _ in range(n)
        ]
    )


class Ticket:
    def __init__(self, id_prefix):
        self.call: Dict[str, Any] = dict()
        self.call["id"] = id_prefix + random_guid()
        self.cancelled: bool = False
        self.processed_by_infmod_guid: str = ""
        self.streaming_queue = asyncio.queues.Queue()

    def id(self):
        return self.call.get("id", None)

    def done(self):
        if "id" in self.call:
            del self.call["id"]
