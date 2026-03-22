# CERT v1：兩條執行路徑

在對接 Wazuh Indexer 之前，可用 **CERT／demo Episode** 驗證 Layer C 與產物目錄。

## 路徑 A：完整 E2E（含 Layer B 推論）

由 `scripts/run_e2e_sample_cert.py` 驅動：CERT 或 demo episode → Layer B job → 推論 → 由推論結果組 Episode + Hypothesis → `src.cli retrieve` / `analyze` / `writeback`。

```bash
cd /path/to/AgenticRAG4TimeSeries
PYTHONPATH=. python scripts/run_e2e_sample_cert.py --episode tests/demo/episode_insider_highrisk.json
```

自 CERT 目錄產生 episodes 後取第一個：

```bash
PYTHONPATH=. python scripts/run_e2e_sample_cert.py --cert-data data --window-days 7
```

產物見腳本結尾說明（`outputs/e2e_sample_cert/`、`outputs/evidence/`、`outputs/agents/`、`outputs/writeback/`）。

## 路徑 B：僅 Layer C（無 Layer B）

已有 Episode JSON（例如 `tests/demo/episode_insider_highrisk.json` 或 `outputs/episodes/cert/*.json`）時，可直接跑管線並可選寫入 `analysis_runs`：

```bash
PYTHONPATH=. python scripts/mvp_cert_layer_c.py --episode tests/demo/episode_insider_highrisk.json
```

等同 C1 retrieve → C2 analyze → C3 writeback（見 `src/pipeline/layer_c_run.run_layer_c_pipeline`）。

## 相關指令（手動分步）

```bash
PYTHONPATH=. python -m src.cli retrieve --episode path/to/episode.json
PYTHONPATH=. python -m src.cli analyze --episode path/to/episode.json
PYTHONPATH=. python -m src.cli writeback --episode path/to/episode.json --mode dry_run
```

若有 Hypothesis 檔，在 `retrieve` / `analyze` 加上 `--hypothesis`。

## 資料集授權

使用官方 CERT 資料集時請遵守其使用條款。
