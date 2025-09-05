from config import MasterConfig

cfg = MasterConfig()

ACTION_HISTORY_LEN = 3

cfg.model.arch = "transformer"
cfg.model.transformer_d_model = 256
cfg.model.transformer_nhead = 8
cfg.model.transformer_layers = 4
cfg.model.transformer_ff = 512
cfg.model.dropout_p = 0.1
cfg.model.additional_feats = 4 + ACTION_HISTORY_LEN * 4

cfg.trainlog.num_val_ep = 3500
cfg.trainlog.val_freq = 1000
cfg.trainlog.episodes = 55_000
cfg.trainlog.plot_top_n = 10

cfg.per.buffer_size = 1_000_000

cfg.rl.batch_size = 16
cfg.rl.learning_rate = 1e-4
cfg.rl.train_start = 10_000

cfg.seq.agent_history_len = 90
cfg.seq.agent_session_len = 60
cfg.seq.action_history_len = ACTION_HISTORY_LEN

cfg.backtest_mode = True
cfg.backtest.max_parallel_sessions = 2
cfg.backtest.position_fraction = 0.5
cfg.backtest.selection_strategy = "advantage_based_filter"
cfg.backtest.long_action_threshold = 0.012695
cfg.backtest.short_action_threshold = 0.009902
cfg.backtest.close_action_threshold = 0.001141
cfg.backtest.ensemble_n_samples = 5
cfg.backtest.ensemble_max_sigma = 0.01
cfg.backtest.return_qvals = True
cfg.backtest.use_cache = True
cfg.backtest.clear_disk_cache = False
cfg.backtest.use_risk_management = False
cfg.backtest.stop_loss = 0.01
cfg.backtest.take_profit = 0.02
cfg.backtest.trailing_stop = 0.005
cfg.backtest.plot_backtest_balance_curve = True

cfg.logging.per_trial_logs = False
cfg.debug.debug_max_size_data = None
cfg.debug.use_final_model = False

# python train.py configs/production.py
# python test_agent.py configs/production.py
# python backtest_engine.py configs/production.py
# python optimize_cfg.py configs/production.py

# Mini run with 10 short sessions
# python optimize_cfg.py configs/production.py --trials 100 --jobs 1

# Notes: Default metric is values_0; default direction is max.
# python get_info_from_optuna.py configs/production.py --n-best-trials 10