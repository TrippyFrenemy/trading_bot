+112
-0

"""Prototype of full production trading system."""
import argparse
import asyncio
import logging
import os
from collections import deque
from typing import Deque, List

import numpy as np
import torch

from agent import D3QN_PER_Agent
from binance_streaming import BinanceDataStream, BinanceOrderExecutor
from utils import load_config

logger = logging.getLogger(__name__)


class ProductionSystem:
    def __init__(self, cfg, symbol: str) -> None:
        self.cfg = cfg
        self.symbol = symbol.upper()
        self.buffer: Deque[List[float]] = deque(maxlen=cfg.seq.agent_history_len)
        self.order_executor = BinanceOrderExecutor()
        self.agent: D3QN_PER_Agent | None = None

    def on_kline(self, kline: dict) -> None:
        feats = [
            float(kline["o"]),
            float(kline["h"]),
            float(kline["v"]),
            float(kline["l"]),
            float(kline["c"]),
            float(kline["q"]),
            float(kline["n"]),
        ]
        self.buffer.append(feats)
        if self.agent and len(self.buffer) == self.buffer.maxlen:
            self.act()

    def init_agent(self) -> None:
        state_shape = (self.cfg.seq.num_features, self.cfg.seq.input_history_len, 1)
        self.agent = D3QN_PER_Agent(
            state_shape=state_shape,
            action_dim=self.cfg.market.num_actions,
            cnn_maps=self.cfg.model.cnn_maps,
            cnn_kernels=self.cfg.model.cnn_kernels,
            cnn_strides=self.cfg.model.cnn_strides,
            dense_val=self.cfg.model.dense_val,
            dense_adv=self.cfg.model.dense_adv,
            additional_feats=self.cfg.model.additional_feats,
            dropout_model=self.cfg.model.dropout_p,
            device=self.cfg.device.device,
            gamma=self.cfg.rl.gamma,
            learning_rate=self.cfg.rl.learning_rate,
            batch_size=self.cfg.rl.batch_size,
            buffer_size=self.cfg.per.buffer_size,
            target_update_freq=self.cfg.rl.target_update_freq,
            train_start=self.cfg.rl.train_start,
            per_alpha=self.cfg.per.per_alpha,
            per_beta_start=self.cfg.per.per_beta_start,
            per_beta_frames=self.cfg.per.per_beta_frames,
            eps_start=self.cfg.eps.eps_start,
            eps_end=self.cfg.eps.eps_end,
            eps_frames=self.cfg.eps.eps_decay_frames,
            epsilon=self.cfg.per.per_eps,
            model_arch=self.cfg.model.arch,
            transformer_kwargs=dict(
                d_model=self.cfg.model.transformer_d_model,
                nhead=self.cfg.model.transformer_nhead,
                num_layers=self.cfg.model.transformer_layers,
                dim_feedforward=self.cfg.model.transformer_ff,
            ),
            max_gradient_norm=self.cfg.rl.max_gradient_norm,
            backtest_cache_path=None,
        )
        model_path = os.path.join(self.cfg.paths.model_dir, "best.pth")
        if os.path.exists(model_path):
            self.agent.load_model(model_path)

    def act(self) -> None:
        assert self.agent is not None
        arr = np.array(self.buffer).T
        state = arr.flatten()
        action = self.agent.select_action(state, training=False)
        qty = self.cfg.backtest.position_fraction
        if action == 1:
            self.order_executor.market_order(self.symbol, "BUY", qty)
        elif action == 2:
            self.order_executor.market_order(self.symbol, "SELL", qty)
        elif action == 3:
            self.order_executor.market_order(self.symbol, "SELL", qty)

    async def run(self) -> None:
        self.init_agent()
        stream = BinanceDataStream(self.symbol, self.on_kline)
        await stream.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run production trading system")
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    system = ProductionSystem(cfg, args.symbol)
    asyncio.run(system.run())


if __name__ == "__main__":
    main()
    