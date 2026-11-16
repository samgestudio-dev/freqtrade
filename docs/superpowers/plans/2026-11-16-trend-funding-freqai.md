# TrendFundingFreqai Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在这个 fork 里新增一个 1h 周期的 FreqAI 策略 `TrendFundingFreqai`（趋势特征 + 资金费率特征 + 三重屏障标签 + LightGBM 分类器），通过偏差检查和滚动回测验收后，以 dry-run 方式与本机现有 bot 并行运行。

**Architecture:** 不改 freqtrade 本体。标签和资金费率特征写成两个纯函数模块（可单测），策略类只负责把它们接进 FreqAI 的 `feature_engineering_*` / `set_freqai_targets`。模型用内置 `LightGBMClassifier`，出场规则与标签的三重屏障保持一致。

**Tech Stack:** freqtrade 2026.11-dev（本 fork 的 `develop`）、FreqAI、LightGBM、pandas、TA-Lib、pytest。

**Spec:** 本文件末尾的「背景与决策」一节（来自 2026-11-16 的调研对话，执行前先读）。

**运行环境:** 策略和模型部署在 Sam 的本地 Mac 上，由本地 Claude Code 执行本计划。本计划写于云端会话，**写的人看不到本机的配置、现有策略和交易对**，所以 Task 1 先盘点本机现状。

## Global Constraints

- 不修改 `freqtrade/` 目录下的任何文件。所有新增代码放在 `user_data/strategies/`。
- 不碰本机正在运行的 bot：不改它的 config、策略文件、数据库，不重启它。新策略用独立的 config 文件、独立的 `--db-url`、独立的 API 端口、独立的 `freqai.identifier`。
- 新 config 必须是 `"dry_run": true`。**把 `dry_run` 改成 `false` 只能由 Sam 本人操作**，本计划的任何步骤都不做这件事。
- 不提交任何含 API key / secret 的文件。`config*.json` 已被 `.gitignore` 忽略，保持这样。
- 需要 `"trading_mode": "futures"`（资金费率只有合约市场有）。交易所按 Binance USDT 本位合约写；Task 1 发现不一致时停下来问 Sam。
- 基础周期 `1h`，杠杆 1x（不重写 `leverage()`）。
- 回测手续费：基准 `--fee 0.001`（每边，含滑点估计），压力测试 `--fee 0.0015`。freqtrade 不模拟滑点，所以不允许用默认手续费出结论。
- 开发区间 `20220601-20260501`；留出区间 `20260501-20261101` **只跑一次**，跑之前不看。
- 调参迭代最多 3 轮，每轮都记进结果文件。3 轮后仍不达标就停下汇报，不继续调。
- pytest 命令统一带 `-o addopts=""`（仓库的 `pyproject.toml` 默认带 xdist 参数，本机不一定装了）。
- 工作分支：`feat/trend-funding-freqai`（本文件所在分支）。每个 Task 结束提交一次。

## Review Focus

以下五种情况最容易在实盘里出问题，每条都已在对应 Task 里配了测试或检查：

1. **同一个币的资金费率结算间隔会变**（8h → 4h → 1h）。特征必须按时间窗口算，不能按事件条数算。→ Task 3 `test_windows_are_time_based_not_event_count`
2. **新上市的币没有或只有很短的资金费率历史**。特征应为 NaN，不能报错，也不能用两三个点算出 z-score。→ Task 3 `test_empty_input_returns_empty_frame_with_columns`、`test_insufficient_history_is_nan`
3. **同一根 K 线同时碰到上下屏障**。1h K 线内部先后顺序未知，标签记 `flat`，不能偏向任一方向。→ Task 2 `test_both_barriers_same_bar_is_flat`
4. **最后 `horizon` 根 K 线的未来还没发生**。标签必须是 NaN，不能是 `flat`，否则实盘重训时会学到假的 `flat`。→ Task 2 `test_tail_rows_are_nan`
5. **实盘能拿到的资金费率历史可能比训练窗口短**（回测从磁盘读完整历史，dry-run/实盘从交易所缓存读）。→ Task 6 Step 3 的 NaN 占比检查

## File Structure

| 文件 | 职责 |
|---|---|
| `.gitignore`（修改，仅在文件末尾追加） | 让 `user_data/strategies/` 进版本控制 |
| `user_data/strategies/tb_labels.py` | `barrier_width`、`triple_barrier_labels` 两个纯函数 |
| `user_data/strategies/funding_feats.py` | `funding_features` 纯函数 |
| `user_data/strategies/TrendFundingFreqai.py` | 策略类 |
| `user_data/strategies/trend_funding.freqai.example.json` | `freqai` 配置块示例（无密钥） |
| `user_data/strategies/tests/conftest.py` | 把上级目录加进 `sys.path` |
| `user_data/strategies/tests/test_tb_labels.py` | 标签单测 |
| `user_data/strategies/tests/test_funding_feats.py` | 资金费率特征单测 |
| `docs/superpowers/plans/trend-funding-results.md` | 本机盘点结果 + 每轮回测记录 |

策略文件能直接 `from tb_labels import ...`：freqtrade 加载策略时会把策略目录临时加进 `sys.path`（`freqtrade/resolvers/iresolver.py` 的 `PathModifier`）。

---

### Task 1: 本机盘点与版本控制准备

**Files:**
- Modify: `.gitignore`（末尾追加）
- Create: `docs/superpowers/plans/trend-funding-results.md`

**Interfaces:**
- Produces: 结果文件里的「本机现状」一节，后续 Task 的交易对列表和 config 路径都从这里取。

- [ ] **Step 1: 切到工作分支**

Run: `git fetch origin && git checkout feat/trend-funding-freqai`
如果本机有未提交改动导致切不过去，停下来告诉 Sam，不要 stash 或丢弃。

- [ ] **Step 2: 盘点本机现状，写进 `docs/superpowers/plans/trend-funding-results.md` 的「本机现状」一节**

要记录的项（只记事实，不记密钥）：
- 正在使用的 config 文件路径、`exchange.name`、`trading_mode`、`timeframe`、`stake_currency`、`dry_run` 的值
- `pair_whitelist` 或 pairlist 方法
- 现有策略的文件名和类名、是否用 FreqAI、用哪个 `--freqaimodel`
- bot 的启动方式（launchd / docker / 终端手动）和 API 端口
- `user_data/data/` 下已有数据的交易所、周期和起始日期（`freqtrade list-data --config <现有config>`）
- `freqtrade --version` 的输出

- [ ] **Step 3: 检查前提，不满足就停下问 Sam**

- `trading_mode` 不是 `futures` → 停。
- `exchange.name` 不是 `binance` → 停。
- `user_data/strategies/` 里已有叫 `tb_labels.py`、`funding_feats.py`、`TrendFundingFreqai.py` 的文件 → 停。

- [ ] **Step 4: 在 `.gitignore` 末尾追加**

```gitignore

# samgestudio fork: track our own strategies
!user_data/strategies/
!user_data/strategies/**
user_data/strategies/**/__pycache__/
```

- [ ] **Step 5: 验证忽略规则**

Run: `mkdir -p user_data/strategies/tests && touch user_data/strategies/tb_labels.py && git check-ignore -q user_data/strategies/tb_labels.py; echo $?`
Expected: `1`（没被忽略）

Run: `git check-ignore -q user_data/config.json; echo $?`
Expected: `0`（仍被忽略）

Run: `git status --short user_data/`
Expected: 只列出本计划要新增的文件。如果列出了本机已有的其他策略文件，**不要 `git add` 它们**，在结果文件里记一笔，交给 Sam 决定。

- [ ] **Step 6: Commit**

```bash
git add .gitignore docs/superpowers/plans/trend-funding-results.md
git commit -m "chore: track user_data/strategies and record local setup"
```

---

### Task 2: 三重屏障标签

**Files:**
- Create: `user_data/strategies/tb_labels.py`
- Create: `user_data/strategies/tests/conftest.py`
- Test: `user_data/strategies/tests/test_tb_labels.py`

**Interfaces:**
- Produces:
  - `barrier_width(close: pd.Series, vol_window: int) -> pd.Series` — `close.pct_change().rolling(vol_window).std()`，索引与输入相同。
  - `triple_barrier_labels(df: pd.DataFrame, width: pd.Series, horizon: int, pt_mult: float, sl_mult: float) -> pd.Series` — `df` 需要 `close`、`high`、`low` 三列；返回 object 类型的 Series，取值 `"up"` / `"down"` / `"flat"` / `NaN`，索引与 `df` 相同。

标签规则（第 t 行）：上屏障 `close[t] * (1 + pt_mult * width[t])`，下屏障 `close[t] * (1 - sl_mult * width[t])`。从 t+1 到 t+horizon 逐根看：`high` 先碰上屏障记 `up`，`low` 先碰下屏障记 `down`，同一根两个都碰记 `flat`，到期都没碰记 `flat`。`width[t]` 为 NaN 或 ≤ 0 时记 NaN。t+horizon 超出数据末尾时记 NaN。

- [ ] **Step 1: 写 `conftest.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

- [ ] **Step 2: 写失败的测试 `test_tb_labels.py`**

```python
import numpy as np
import pandas as pd

from tb_labels import barrier_width, triple_barrier_labels


def _df(high, low, close=None):
    n = len(high)
    return pd.DataFrame({"close": close or [100.0] * n, "high": high, "low": low})


def _labels(df, width=0.02, horizon=3):
    w = pd.Series([width] * len(df)) if np.isscalar(width) else pd.Series(width)
    return triple_barrier_labels(df, w, horizon=horizon, pt_mult=1.0, sl_mult=1.0)


def test_upper_barrier_hit_is_up():
    # row 0: barriers 102 / 98; bar 2 high 103 touches upper
    df = _df(high=[100, 101, 103, 100, 100], low=[100, 99, 99, 99, 99])
    assert _labels(df).iloc[0] == "up"


def test_lower_barrier_hit_is_down():
    df = _df(high=[100, 100, 100, 100, 100], low=[100, 99, 97, 99, 99])
    assert _labels(df).iloc[0] == "down"


def test_first_touch_wins():
    # bar 1 touches lower, bar 2 touches upper
    df = _df(high=[100, 100, 103, 100, 100], low=[100, 97, 99, 99, 99])
    assert _labels(df).iloc[0] == "down"


def test_both_barriers_same_bar_is_flat():
    df = _df(high=[100, 103, 100, 100, 100], low=[100, 97, 100, 100, 100])
    assert _labels(df).iloc[0] == "flat"


def test_no_touch_within_horizon_is_flat():
    df = _df(high=[100, 101, 101, 101, 105], low=[100, 99, 99, 99, 99])
    # bar 4 is outside row 0's horizon of 3
    assert _labels(df).iloc[0] == "flat"


def test_tail_rows_are_nan():
    df = _df(high=[100] * 5, low=[100] * 5)
    labels = _labels(df, horizon=3)
    assert labels.iloc[:2].notna().all()
    assert labels.iloc[2:].isna().all()


def test_nan_or_zero_width_is_nan():
    df = _df(high=[100, 103, 100, 100, 100], low=[100, 99, 99, 99, 99])
    labels = _labels(df, width=[np.nan, 0.0, 0.02, 0.02, 0.02])
    assert pd.isna(labels.iloc[0])
    assert pd.isna(labels.iloc[1])


def test_index_is_preserved():
    df = _df(high=[100] * 5, low=[100] * 5)
    df.index = [10, 11, 12, 13, 14]
    w = pd.Series([0.02] * 5, index=df.index)
    labels = triple_barrier_labels(df, w, horizon=3, pt_mult=1.0, sl_mult=1.0)
    assert list(labels.index) == [10, 11, 12, 13, 14]


def test_barrier_width_is_rolling_std_of_returns():
    close = pd.Series([100.0, 101.0, 100.0, 102.0, 101.0])
    expected = close.pct_change().rolling(3).std()
    pd.testing.assert_series_equal(barrier_width(close, 3), expected)
```

- [ ] **Step 3: 运行，确认失败**

Run: `python -m pytest user_data/strategies/tests/test_tb_labels.py -o addopts="" -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'tb_labels'`（或 ImportError）

- [ ] **Step 4: 在 `user_data/strategies/tb_labels.py` 实现两个函数**

签名见上面的 Interfaces。`triple_barrier_labels` 用 numpy 数组逐行向前扫描即可（1h 数据量小，不需要向量化）。

- [ ] **Step 5: 运行，确认通过**

Run: `python -m pytest user_data/strategies/tests/test_tb_labels.py -o addopts="" -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add user_data/strategies/tb_labels.py user_data/strategies/tests/
git commit -m "feat: add triple-barrier labels"
```

---

### Task 3: 资金费率特征

**Files:**
- Create: `user_data/strategies/funding_feats.py`
- Test: `user_data/strategies/tests/test_funding_feats.py`

**Interfaces:**
- Produces:
  - `FUNDING_COLUMNS = ["fr", "fr_cum_7d", "fr_z_30d", "fr_mom_7d", "fr_extreme"]`
  - `EXTREME_FUNDING = 0.0005`
  - `funding_features(funding: pd.DataFrame) -> pd.DataFrame` — 输入列 `date`（UTC 时区的 datetime）和 `funding_rate`，每个结算事件一行。输出列为 `date` + `FUNDING_COLUMNS`，行数与输入相同。

各列定义（都只用当前行及之前的数据）：

| 列 | 定义 |
|---|---|
| `fr` | 当次资金费率 |
| `fr_cum_7d` | 过去 7 天（时间窗口 `(t-7D, t]`）的费率之和 |
| `fr_z_30d` | `(fr - 30天均值) / 30天标准差`；标准差小于 `1e-12` 时取 0 |
| `fr_mom_7d` | 过去 1 天均值 − 过去 7 天均值 |
| `fr_extreme` | `abs(fr) > EXTREME_FUNDING` 时为 1，否则 0 |

历史不足时为 NaN：`fr_cum_7d` 和 `fr_mom_7d` 在距首个事件不满 7 天时为 NaN，`fr_z_30d` 在不满 30 天时为 NaN。

- [ ] **Step 1: 写失败的测试 `test_funding_feats.py`**

```python
import pandas as pd
import pytest

from funding_feats import EXTREME_FUNDING, FUNDING_COLUMNS, funding_features


def _events(start, periods, freq, rate=0.0001):
    dates = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame({"date": dates, "funding_rate": rate})


def test_output_shape_and_columns():
    f = _events("2025-01-01", 120, "8h")
    out = funding_features(f)
    assert list(out.columns) == ["date", *FUNDING_COLUMNS]
    assert len(out) == len(f)


def test_constant_rate_values():
    out = funding_features(_events("2025-01-01", 120, "8h"))  # 40 days
    last = out.iloc[-1]
    assert last["fr"] == pytest.approx(0.0001)
    assert last["fr_cum_7d"] == pytest.approx(21 * 0.0001)  # 3 events/day * 7
    assert last["fr_z_30d"] == pytest.approx(0.0)  # zero std -> 0, not NaN/inf
    assert last["fr_mom_7d"] == pytest.approx(0.0)
    assert last["fr_extreme"] == 0


def test_windows_are_time_based_not_event_count():
    # 20 days at 8h, then 20 days at 4h
    a = _events("2025-01-01", 60, "8h")
    b = _events("2025-01-21", 120, "4h")
    out = funding_features(pd.concat([a, b], ignore_index=True))
    assert out.iloc[-1]["fr_cum_7d"] == pytest.approx(42 * 0.0001)  # 6 events/day * 7


def test_extreme_flag_is_strictly_greater_and_symmetric():
    f = _events("2025-01-01", 3, "8h")
    f["funding_rate"] = [EXTREME_FUNDING, 0.0006, -0.0006]
    assert funding_features(f)["fr_extreme"].tolist() == [0, 1, 1]


def test_insufficient_history_is_nan():
    out = funding_features(_events("2025-01-01", 120, "8h"))
    day3 = out[out["date"] == pd.Timestamp("2025-01-04", tz="UTC")].iloc[0]
    day10 = out[out["date"] == pd.Timestamp("2025-01-11", tz="UTC")].iloc[0]
    assert pd.isna(day3["fr_cum_7d"]) and pd.isna(day3["fr_mom_7d"])
    assert pd.notna(day10["fr_cum_7d"]) and pd.isna(day10["fr_z_30d"])
    assert pd.notna(out.iloc[-1]["fr_z_30d"])


def test_no_lookahead():
    f = _events("2025-01-01", 120, "8h")
    f["funding_rate"] = [0.0001 * ((i % 7) - 3) for i in range(120)]
    full = funding_features(f)
    truncated = funding_features(f.iloc[:100])
    pd.testing.assert_frame_equal(full.iloc[:100], truncated)


def test_empty_input_returns_empty_frame_with_columns():
    out = funding_features(pd.DataFrame({"date": [], "funding_rate": []}))
    assert list(out.columns) == ["date", *FUNDING_COLUMNS]
    assert len(out) == 0
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest user_data/strategies/tests/test_funding_feats.py -o addopts="" -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'funding_feats'`

- [ ] **Step 3: 在 `user_data/strategies/funding_feats.py` 实现**

用 pandas 的时间窗口滚动（`rolling("7D", on="date")`），再按「距首个事件的时长」把历史不足的行置为 NaN。

- [ ] **Step 4: 运行，确认通过**

Run: `python -m pytest user_data/strategies/tests -o addopts="" -v`
Expected: 16 passed（Task 2 的 9 个 + 本 Task 的 7 个）

- [ ] **Step 5: Commit**

```bash
git add user_data/strategies/funding_feats.py user_data/strategies/tests/test_funding_feats.py
git commit -m "feat: add funding-rate features"
```

---

### Task 4: 策略类与配置

**Files:**
- Create: `user_data/strategies/TrendFundingFreqai.py`
- Create: `user_data/strategies/trend_funding.freqai.example.json`
- Create（本机，不提交）: `user_data/config_trend_funding.json`

**Interfaces:**
- Consumes: Task 2 的 `barrier_width`、`triple_barrier_labels`；Task 3 的 `funding_features`、`FUNDING_COLUMNS`。
- Produces: 策略类 `TrendFundingFreqai`；标签列 `&-tb_label`；分类器输出的概率列 `up` / `down` / `flat`。

先读这三处再动手：`freqtrade/templates/FreqaiExampleStrategy.py`（FreqAI 策略骨架）、`tests/strategy/strats/freqai_test_classifier.py`（分类器的 `class_names` 用法）、`docs/strategy-customization.md` 的 "Historic funding rate data" 一节（资金费率必须按日期合并）。

- [ ] **Step 1: 写 `trend_funding.freqai.example.json`**

```json
{
    "freqai": {
        "enabled": true,
        "identifier": "trend_funding_v1",
        "purge_old_models": 2,
        "train_period_days": 180,
        "backtest_period_days": 30,
        "live_retrain_hours": 24,
        "feature_parameters": {
            "include_timeframes": ["1h", "4h"],
            "include_corr_pairlist": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
            "label_period_candles": 24,
            "include_shifted_candles": 1,
            "indicator_periods_candles": [24, 72, 168],
            "DI_threshold": 0,
            "principal_component_analysis": false,
            "use_SVM_to_remove_outliers": false,
            "plot_feature_importances": 20
        },
        "data_split_parameters": {
            "test_size": 0.2,
            "shuffle": false
        },
        "model_training_parameters": {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 100,
            "colsample_bytree": 0.5,
            "verbosity": -1
        }
    }
}
```

- [ ] **Step 2: 实现 `TrendFundingFreqai(IStrategy)`**

类属性（值都是定死的）：

| 属性 | 值 |
|---|---|
| `timeframe` | `"1h"` |
| `can_short` | `True` |
| `process_only_new_candles` | `True` |
| `startup_candle_count` | `168` |
| `minimal_roi` | `{"0": 10}`（等于关闭 ROI，出场全靠屏障） |
| `stoploss` | `-0.10`（兜底，正常由 `custom_stoploss` 接管） |
| `use_custom_stoploss` | `True` |
| `use_exit_signal` | `True` |
| `PT_MULT` / `SL_MULT` | `3.0` / `3.0` |
| `VOL_WINDOW` | `72` |
| `ENTRY_PROB` | `0.55` |
| `FALLBACK_WIDTH` | `0.03` |

方法：

- `informative_pairs()` → 对白名单里每个交易对返回 `(pair, "1h", "funding_rate")`，保证 dry-run/实盘会刷新资金费率。
- `feature_engineering_expand_all(dataframe, period, metadata, **kwargs)` → 5 个特征：`%-roc-period`（`close.pct_change(period)`）、`%-ema_dist-period`（`close / EMA(period) - 1`）、`%-adx-period`、`%-natr-period`、`%-rel_volume-period`（`volume / volume.rolling(period).mean()`）。
- `feature_engineering_expand_basic(dataframe, metadata, **kwargs)` → `%-ret_1`（`close.pct_change()`）。
- `feature_engineering_standard(dataframe, metadata, **kwargs)` → `%-hour_of_day`、`%-day_of_week`，再调用 `_merge_funding(dataframe, metadata["pair"], prefix="%-")`。
- `_merge_funding(dataframe, pair, prefix) -> DataFrame` → `self.dp.get_pair_dataframe(pair=pair, timeframe="1h", candle_type="funding_rate")` 取事件级数据，过 `funding_features()`，把 `FUNDING_COLUMNS` 重命名为 `f"{prefix}{col}"` 后用 `merge_informative_pair(..., ffill=True)` 按日期并入，最后删掉合并产生的多余日期列。**先在事件级算完特征再合并**，不要先 ffill 再滚动求和。
- `set_freqai_targets(dataframe, metadata, **kwargs)` → `self.freqai.class_names = ["down", "flat", "up"]`；`horizon` 取 `self.freqai_info["feature_parameters"]["label_period_candles"]`；`dataframe["&-tb_label"] = triple_barrier_labels(dataframe, barrier_width(dataframe["close"], self.VOL_WINDOW), horizon, self.PT_MULT, self.SL_MULT)`。
- `populate_indicators(dataframe, metadata)` → `dataframe = self.freqai.start(dataframe, metadata, self)`；然后加两组非特征列：`tb_width = barrier_width(close, VOL_WINDOW)`，以及 `_merge_funding(dataframe, metadata["pair"], prefix="")` 得到的 `fr`、`fr_extreme` 等。
- `populate_entry_trend(df, metadata)`:
  - 做多：`do_predict == 1` 且 `&-tb_label == "up"` 且 `up >= ENTRY_PROB` 且 `volume > 0` 且**不是**（`fr_extreme == 1` 且 `fr > 0`）。`enter_tag = "tb_long"`。
  - 做空：镜像条件，`"down"`、`down >= ENTRY_PROB`、**不是**（`fr_extreme == 1` 且 `fr < 0`）。`enter_tag = "tb_short"`。
  - `fr` 为 NaN 时不拦截（视为非极端）。
- `populate_exit_trend(df, metadata)` → 原样返回 `df`。
- `_entry_width(pair, trade) -> float` → 按 `docs/strategy-callbacks.md` 里的写法，用 `timeframe_to_prev_date(self.timeframe, trade.open_date_utc)` 在已分析的 dataframe 里找开仓那根 K 线的**前一根**（信号 K 线），返回它的 `tb_width`；找不到或为 NaN 时返回 `FALLBACK_WIDTH`。
- `custom_stoploss(...)` → 用 `stoploss_from_open(-SL_MULT * width, current_profit, is_short=trade.is_short, leverage=trade.leverage)`。
- `custom_exit(...)` → `current_profit >= PT_MULT * width` 返回 `"tp_barrier"`；持仓时长 ≥ `horizon` 小时返回 `"time_barrier"`；否则 `None`。

- [ ] **Step 3: 在本机生成 `user_data/config_trend_funding.json`（不提交）**

复制 Task 1 记录的现有 config，然后：
- 把 `freqai` 整块换成 Step 1 的内容
- `"dry_run": true`
- `"timeframe": "1h"`、`"trading_mode": "futures"`、`"margin_mode": "isolated"`
- `"api_server.listen_port"` 换一个与现有 bot 不同的端口
- `"bot_name": "trend-funding-dryrun"`
- 白名单：Task 1 记录的交易对里，**在 2021-09-01 之前就已上市**的那些，再确保包含 `BTC/USDT:USDT` 和 `ETH/USDT:USDT`。不足 6 个时停下问 Sam。

Run: `git check-ignore -q user_data/config_trend_funding.json; echo $?`
Expected: `0`

- [ ] **Step 4: 下载数据**

```bash
freqtrade download-data --config user_data/config_trend_funding.json \
  --timeframes 1h 4h --timerange 20210901-20261115
```

合约模式会自动带上 `funding_rate`。如果本机已有的数据起点晚于 2021-09-01，上面的命令只会往后追加，需要再跑一次同样的命令并加 `--prepend` 往前补（`--prepend` 模式下不会追加新数据，所以两次都要跑）。完成后：

Run: `freqtrade list-data --config user_data/config_trend_funding.json --show-timerange`
Expected: 白名单里每个交易对都有 `1h`、`4h` 的 futures 数据和 funding_rate 数据，起点不晚于 2021-09-30。把输出贴进结果文件。

- [ ] **Step 5: 冒烟回测（两个月，只看能不能跑通）**

```bash
freqtrade backtesting --config user_data/config_trend_funding.json \
  --strategy TrendFundingFreqai --freqaimodel LightGBMClassifier \
  --timerange 20240101-20240301 --fee 0.001
```

Expected: 正常结束、无异常栈；日志里没有 `No data found for (…, funding_rate)`；结果表里 exit reason 只出现 `tp_barrier`、`time_barrier`、`trailing_stop_loss` / `stop_loss`、`force_exit`。交易数为 0 也算跑通，但要记进结果文件。

- [ ] **Step 6: Commit**

```bash
git add user_data/strategies/TrendFundingFreqai.py user_data/strategies/trend_funding.freqai.example.json docs/superpowers/plans/trend-funding-results.md
git commit -m "feat: add TrendFundingFreqai strategy"
```

---

### Task 5: 偏差检查与滚动回测验收

**Files:**
- Modify: `docs/superpowers/plans/trend-funding-results.md`

**Interfaces:**
- Consumes: Task 4 的策略和本机 config。
- Produces: 结果文件里的「验收」一节，结论是 PASS 或 FAIL。

所有命令都带 `--config user_data/config_trend_funding.json --strategy TrendFundingFreqai --freqaimodel LightGBMClassifier`，下面简写为 `$ARGS`。

- [ ] **Step 1: 未来数据检查**

Run: `freqtrade lookahead-analysis $ARGS --timerange 20240101-20240601`
Expected: `has_bias` 为 `False`（或输出 "no bias detected"）。发现偏差 → 修代码，不进入下一步。

- [ ] **Step 2: 递归指标检查**

Run: `freqtrade recursive-analysis $ARGS --timerange 20240101-20240301 --startup-candle 168 336 672`
Expected: 各指标差异 < 0.1%。超过 → 调大 `startup_candle_count` 后重跑，并在结果文件里记下新值。

- [ ] **Step 3: 开发区间回测（基准手续费）**

Run: `freqtrade backtesting $ARGS --timerange 20220601-20260501 --fee 0.001 --breakdown year --export trades`
把汇总表和按年拆分表贴进结果文件。

- [ ] **Step 4: 开发区间回测（压力手续费）**

Run: `freqtrade backtesting $ARGS --timerange 20220601-20260501 --fee 0.0015 --breakdown year`

- [ ] **Step 5: 对照验收门槛**

| 门槛 | 要求 |
|---|---|
| 交易数（Step 3） | ≥ 200 |
| Profit factor（Step 3） | ≥ 1.15 |
| 最大回撤（Step 3） | ≤ 25% |
| 按年盈亏（Step 3） | 2022、2023、2024、2025、2026 五段里至少 3 段为正 |
| Profit factor（Step 4） | ≥ 1.0 |

全部满足 → Step 7。有任何一项不满足 → Step 6。

- [ ] **Step 6: 不达标时的迭代（最多 3 轮）**

每轮只改一类东西，按这个顺序试：① `ENTRY_PROB`（0.55 → 0.60）；② `PT_MULT` / `SL_MULT`（3.0 → 2.5 或 3.5，两个一起改）；③ `train_period_days`（180 → 365）。每轮重跑 Step 3–5，把改了什么和结果记进结果文件。**不要**增加特征、换模型或调 LightGBM 参数。3 轮后仍不达标：结论记 FAIL，停止，向 Sam 汇报，不进入 Task 6。

- [ ] **Step 7: 留出区间回测（只跑一次）**

Run: `freqtrade backtesting $ARGS --timerange 20260501-20261101 --fee 0.001 --breakdown month`
Expected: Profit factor ≥ 1.0 且最大回撤 ≤ 25% → 结论 PASS。否则结论 FAIL，停止并汇报，不回头再调参。

- [ ] **Step 8: Commit**

```bash
git add user_data/strategies docs/superpowers/plans/trend-funding-results.md
git commit -m "docs: record TrendFundingFreqai validation results"
git push -u origin feat/trend-funding-freqai
```

---

### Task 6: 并行 dry-run（仅在 Task 5 结论为 PASS 时执行）

**Files:**
- Modify: `docs/superpowers/plans/trend-funding-results.md`

**Interfaces:**
- Consumes: Task 5 的 PASS 结论。

- [ ] **Step 1: 确认 config 仍是 dry-run**

Run: `python -c "import json;print(json.load(open('user_data/config_trend_funding.json'))['dry_run'])"`
Expected: `True`

- [ ] **Step 2: 与现有 bot 并行启动**

```bash
freqtrade trade --config user_data/config_trend_funding.json \
  --strategy TrendFundingFreqai --freqaimodel LightGBMClassifier \
  --db-url sqlite:///user_data/tradesv3.trend_funding.dryrun.sqlite \
  --logfile user_data/logs/trend_funding.log
```

启动方式沿用 Task 1 记录的现有 bot 的方式（launchd / docker / 终端）。启动后确认现有 bot 仍在运行、端口没冲突。

- [ ] **Step 3: 检查实盘路径下的资金费率历史够不够长**

第一次模型训练完成后，在日志里找每个交易对的 `dropped N training points due to NaNs in populated dataset M`，算出 N / M。同时用 freqtrade 的 API 或 FreqUI 看任意一个交易对最新一根 K 线的 `fr_z_30d` 是否有值。

Expected: N / M < 5%，日志里没有 `percent of training data dropped due to NaNs` 的警告，且 `fr_z_30d` 不是 NaN。
不满足 → 说明 dry-run 下拿到的资金费率历史比训练窗口短。**停掉这个 dry-run 进程，记录现象，向 Sam 汇报**，不要自行改成别的数据读取方式。

- [ ] **Step 4: 写下停用规则并收尾**

在结果文件里记录启动时间，并写明：dry-run 至少跑 30 天；30 天内回撤超过 15% 或连续 14 天无交易，就停掉并汇报。是否切实盘、用多少资金，由 Sam 决定。

```bash
git add docs/superpowers/plans/trend-funding-results.md
git commit -m "docs: record dry-run start for TrendFundingFreqai"
git push
```

---

## 背景与决策

### 为什么是这个方向

个人项目在速度、数据和算力上比不过机构，所以选的是对延迟不敏感、数据免费、模型简单的方向：

- **周期选 1h**。订单簿失衡类信号在秒级就被消化，按 K 线运行的 freqtrade 吃不到；小时到日级的时序趋势在计入真实成本后证据仍然较强。
- **模型选 LightGBM，不选深度模型**。2025 年一篇用 Bybit 订单簿数据的对比研究发现，做好预处理后简单模型能追平甚至超过 DeepLOB 这类网络。时序基础模型（Chronos、TimesFM）零样本预测 BTC 的效果不如 ARIMA。
- **资金费率当特征和过滤条件，不当独立策略**。主流币的资金费率套利已被机构资金压平；极端资金费率的反向信号在 BTC 上逐年衰减，只在高波动行情里还有效。
- **标签用三重屏障，不用固定周期收益**。止盈、止损、到期三道屏障更接近真实交易的结果；出场规则与标签一致，模型学的和实际执行的是同一件事。
- **验证比模型重要**。未来数据检查、递归指标检查、含熊市的回测区间、计入滑点的手续费、预先写死的验收门槛和留出区间，都是为了不被过拟合的回测骗到。

### 这一期不做的事

| 不做 | 原因 |
|---|---|
| 持仓量特征 | 本 fork 已原生支持 `open_interest` K 线类型，但 Binance 只提供最近 30 天历史，不够回测。要做需从 data.binance.vision 的 `metrics` 文件补历史，或换 Bybit |
| 爆仓数据特征 | freqtrade 无原生支持，需自建拉取进程 |
| 元标注（第二个模型定仓位） | 需要自定义 `user_data/freqaimodels/` 模型类；等基线验收通过再加 |
| 低成交量山寨币的短期反转 | 研究上有依据，但滑点和容量要单独建模；作为下一个独立计划 |
| Purged / Combinatorial CV | FreqAI 内置的是滑动窗口重训；CPCV 适合放在离线 notebook 里做研究 |
| Hyperopt | 参数搜索越有效越容易过拟合；本计划只允许 3 轮手动迭代 |

### 需要知道的风险

- 引用的学术研究样本多数截至 2022 年前后，资金费率相关的几条来自个人博客。它们决定了方向，不保证结果，所以才有 Task 5 的验收门槛。
- 通过回测验收不等于实盘盈利。Task 6 的 dry-run 是必经步骤，不是形式。

### 参考资料

- [Exploring Microstructural Dynamics in Cryptocurrency Limit Order Books (arXiv 2506.05764)](https://arxiv.org/abs/2506.05764)
- [Time-Series and Cross-Sectional Momentum in the Cryptocurrency Market (Han, Kang, Ryu)](https://acfr.aut.ac.nz/__data/assets/pdf_file/0009/918729/Time_Series_and_Cross_Sectional_Momentum_in_the_Cryptocurrency_Market_with_IA.pdf)
- [A Trend Factor for the Cross Section of Cryptocurrency Returns (JFQA)](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/S0022109024000747)
- [Trading volume and liquidity provision in cryptocurrency markets (Riksbank WP 413)](https://www.riksbank.se/globalassets/media/rapporter/working-papers/2022/no.-413-trading-volume-and-liquidity-provision-in-cryptocurreny-markets.pdf)
- [Crypto Carry: The Funding-Rate Cross-Section](https://bryanvine.github.io/alpha-research/paper2.html)
- [The Crypto Carry Trade is Dead (BitMEX)](https://www.bitmex.com/blog/crypto-carry-trade-death)
- [Funding-Crowding Edge Survives Only in Panic Regimes](https://askrubyai.github.io/blog/posts/2026-02-26-funding-regime-filter/index.html)
- [When a Model Trained on Everything Fails on Bitcoin](https://jeromeetienne.github.io/blog/2026-06-15-post-06-pretrained.html)
- [Purged Cross-validation (LuxAlgo)](https://www.luxalgo.com/library/concept/purged-cross-validation.md)
- 本仓库文档：`docs/freqai-running.md`、`docs/freqai-feature-engineering.md`、`docs/strategy-customization.md`（Historic funding rate data）、`docs/lookahead-analysis.md`、`docs/recursive-analysis.md`
