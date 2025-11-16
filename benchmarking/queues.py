import asyncio

class DropOldestQueue(asyncio.Queue):
    """Keeps latency bounded: if full, drops one oldest item."""
    async def put(self, item):
        if self.full():
            try:
                self.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await super().put(item)
