"""Utilities for real-time Binance data and order execution."""
import asyncio
import logging
import os
from typing import Any, Callable, Dict

from binance import AsyncClient, BinanceSocketManager, Client

logger = logging.getLogger(__name__)


class BinanceDataStream:
    """Subscribe to 1m kline updates via WebSocket."""

    def __init__(self, symbol: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        self.symbol = symbol.lower()
        self.callback = callback
        self._client: AsyncClient | None = None
        self._manager: BinanceSocketManager | None = None

    async def connect(self) -> None:
        self._client = await AsyncClient.create()
        self._manager = BinanceSocketManager(self._client)
        self._socket = self._manager.kline_socket(symbol=self.symbol, interval="1m")

    async def run(self) -> None:
        if self._client is None:
            await self.connect()
        assert self._socket is not None
        async with self._socket as stream:
            while True:
                msg = await stream.recv()
                if msg and msg.get("e") == "kline":
                    self.callback(msg["k"])

    async def close(self) -> None:
        if self._client:
            await self._client.close_connection()


class BinanceOrderExecutor:
    """Execute orders on Binance Futures via REST."""

    def __init__(self) -> None:
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        self.client = Client(api_key, api_secret)

    def market_order(self, symbol: str, side: str, quantity: float) -> Dict[str, Any]:
        logger.info("Placing %s order for %s qty %s", side, symbol, quantity)
        return self.client.create_order(symbol=symbol, side=side, type="MARKET", quantity=quantity)
    