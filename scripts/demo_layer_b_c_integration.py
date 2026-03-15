#!/usr/bin/env python3
"""
Layer B / Layer C 測試展示腳本：啟動 Control Plane（可選）→ Layer B（inference + POST）→ Layer C（retrieve → analyze → writeback + POST）→ 串通成功。

顯示每步資料處理經過，最後以 writeback 已寫入並送至 Control Plane 收尾，輸出「串通成功」。

Usage:
  PYTHONPATH=. python scripts/demo_layer_b_c_integration.py --no-start-cp
  PYTHONPATH=. python scripts/demo_layer_b_c_integration.py --episode tests/demo/episode_insider_highrisk.json
  PYTHONPATH=. python scripts/demo_layer_b_c_integration.py --control-plane-url http://127.0.0.1:8081 --control-plane-token demo-token
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _health_check(base_url: str, timeout: int = 5) -> bool:
    url = f"{base_url.rstrip('/')}/health"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _wait_health(base_url: str, max_wait_sec: float = 60, interval: float = 1.0) -> bool:
    deadline = time.monotonic() + max_wait_sec
    while time.monotonic() < deadline:
        if _health_check(base_url, timeout=3):
            return True
        time.sleep(interval)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Layer B/C 測試展示：Control Plane → Layer B (POST) → Layer C (writeback + POST) → 串通成功"
    )
    parser.add_argument("--episode", type=str, help="Episode JSON 路徑（如 tests/demo/episode_insider_highrisk.json）")
    parser.add_argument("--cert-data", type=str, default="data", help="無 --episode 時從此目錄跑 cert2episodes")
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--control-plane-url", type=str, default="http://127.0.0.1:8081", help="Control Plane 基底 URL")
    parser.add_argument("--control-plane-token", type=str, default="demo-token", help="Control Plane Bearer token")
    parser.add_argument("--no-start-cp", action="store_true", help="不啟動 Control Plane，僅檢查 health 後跑流程")
    parser.add_argument("--guacamole-root", type=str, default=None, help="Guacamole 專案根目錄（預設 REPO_ROOT/../guacamole-ai）")
    parser.add_argument("--out-dir", type=str, default="outputs/demo_layer_b_c", help="輸出目錄")
    args = parser.parse_args()

    base_url = args.control_plane_url.rstrip("/")
    token = args.control_plane_token or "demo-token"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "CONTROL_PLANE_BASE_URL": base_url, "CONTROL_PLANE_TOKEN": token}
    os.environ["CONTROL_PLANE_BASE_URL"] = base_url
    os.environ["CONTROL_PLANE_TOKEN"] = token

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cp_proc = None

    # ----- Step 0: 檢查 / 啟動 Control Plane -----
    print("\n[Step 0] Control Plane")
    print(f"  CONTROL_PLANE_BASE_URL = {base_url}")
    if not args.no_start_cp:
        guacamole_root = Path(args.guacamole_root) if args.guacamole_root else REPO_ROOT.parent / "guacamole-ai"
        if not guacamole_root.is_dir():
            print(f"  錯誤：Guacamole 目錄不存在: {guacamole_root}，請用 --no-start-cp 並手動啟動 Control Plane。", file=sys.stderr)
            return 1
        print(f"  啟動 Guacamole Control Plane（cwd={guacamole_root}）...")
        cp_proc = subprocess.Popen(
            [sys.executable, "-m", "sensel_control_plane.main"],
            cwd=str(guacamole_root),
            env={**env, "PYTHONPATH": str(guacamole_root)},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if not _wait_health(base_url, max_wait_sec=45):
            if cp_proc.poll() is None:
                cp_proc.terminate()
            err = cp_proc.stderr.read().decode("utf-8", errors="replace") if cp_proc.stderr else ""
            print(f"  health 檢查失敗。stderr: {err[:500]}", file=sys.stderr)
            return 1
        print("  health 狀態: OK")
    else:
        if not _health_check(base_url):
            print("  health 狀態: 失敗（請先啟動 Control Plane 或檢查 --control-plane-url）", file=sys.stderr)
            return 1
        print("  health 狀態: OK")

    try:
        # ----- 解析 episode 來源 -----
        episode_path: Path | None = None
        if args.episode:
            episode_path = REPO_ROOT / args.episode
            if not episode_path.exists():
                print(f"Episode 不存在: {episode_path}", file=sys.stderr)
                return 1
        else:
            print("Running cert2episodes...")
            r = subprocess.run(
                [
                    sys.executable, "-m", "src.cli", "cert2episodes",
                    "--data_dir", args.cert_data, "--out_dir", "outputs/episodes/cert",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                env=env,
            )
            if r.returncode != 0:
                print("cert2episodes 失敗:", r.stderr or r.stdout, file=sys.stderr)
                return 1
            episodes_dir = REPO_ROOT / "outputs" / "episodes" / "cert"
            eps = list(episodes_dir.glob("*.json")) if episodes_dir.exists() else []
            if not eps:
                print("cert2episodes 後無 episode JSON", file=sys.stderr)
                return 1
            episode_path = eps[0]

        ep_data = json.loads(episode_path.read_text(encoding="utf-8"))
        episode_id = ep_data.get("episode_id", "ep-unknown")
        events = ep_data.get("events", [])
        t0_ms = ep_data.get("t0_ms", 0)
        t1_ms = ep_data.get("t1_ms", 0)
        entities = ep_data.get("entities", [])
        entity_primary = entities[0] if entities else "unknown"

        # ----- Step 1: 準備 Layer B job -----
        print("\n[Step 1] 準備 Layer B job")
        job_data = {
            "request": {
                "request_id": f"req-{episode_id}",
                "job_id": f"job-{episode_id}",
                "tenant_id": ep_data.get("tenant_id", "tenant-default"),
                "endpoint_id": entity_primary,
                "t0_ms": t0_ms,
                "t1_ms": t1_ms,
            },
            "events": events,
        }
        job_path = out_dir / "layerb_job.json"
        job_path.write_text(json.dumps(job_data, indent=2), encoding="utf-8")
        print(f"  job 路徑: {job_path.relative_to(REPO_ROOT)}")
        print(f"  episode_id: {episode_id}, endpoint_id: {entity_primary}, 時間窗: t0_ms={t0_ms}, t1_ms={t1_ms}")

        # ----- Step 2: Layer B inference（不 --post），再由腳本 POST -----
        print("\n[Step 2] Layer B inference + POST inference result → Control Plane")
        r = subprocess.run(
            [sys.executable, "scripts/run_layerb_job.py", "--job", str(job_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        if r.returncode != 0:
            print("  Layer B 失敗:", r.stderr or r.stdout, file=sys.stderr)
            return 1
        result_data = json.loads(r.stdout)
        result_path = out_dir / "layerb_result.json"
        result_path.write_text(json.dumps(result_data, indent=2), encoding="utf-8")

        from contracts import InferenceResult
        result = InferenceResult.model_validate(result_data)
        from connectors.control_plane_client import post_result, save_outbox
        ok, err = post_result(result)
        if not ok:
            save_outbox(result, err or "unknown")
            print(f"  POST inference result 失敗: {err}", file=sys.stderr)
            return 1
        risk = result_data.get("hypothesis", {}).get("risk_score")
        hyp_id = result_data.get("hypothesis", {}).get("hypothesis_id", "")
        print(f"  POST inference result → Control Plane: 成功 (hypothesis_id={hyp_id}, risk_score={risk})")

        # ----- Step 3: Build Episode + Hypothesis -----
        print("\n[Step 3] Build Episode + Hypothesis")
        from src.pipeline.build_episode_from_inference import run_build_episode_from_inference
        ep_out, hyp_out = run_build_episode_from_inference(
            result_path,
            job_path=job_path,
            episode_path=out_dir / f"{episode_id}.json",
            hypothesis_path=out_dir / f"{episode_id}_hypothesis.json",
            episode_id=episode_id,
            repo_root=REPO_ROOT,
        )
        print(f"  episode: {ep_out.relative_to(REPO_ROOT)}, hypothesis: {hyp_out.relative_to(REPO_ROOT)}")

        # ----- Step 4: Layer C retrieve -----
        print("\n[Step 4] Layer C retrieve")
        r = subprocess.run(
            [sys.executable, "-m", "src.cli", "retrieve", "--episode", str(ep_out), "--hypothesis", str(hyp_out)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        if r.returncode != 0:
            print("  retrieve 失敗:", r.stderr or r.stdout, file=sys.stderr)
            return 1
        evidence_path = REPO_ROOT / "outputs" / "evidence" / f"{episode_id}.json"
        evidence_count = ""
        if evidence_path.exists():
            try:
                ev = json.loads(evidence_path.read_text(encoding="utf-8"))
                items = ev.get("items", [])
                evidence_count = f", evidence 筆數: {len(items)}"
            except Exception:
                pass
        print(f"  retrieve 完成{evidence_count}")

        # ----- Step 5: Layer C analyze -----
        print("\n[Step 5] Layer C analyze")
        r = subprocess.run(
            [sys.executable, "-m", "src.cli", "analyze", "--episode", str(ep_out), "--hypothesis", str(hyp_out)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        if r.returncode != 0:
            print("  analyze 失敗:", r.stderr or r.stdout, file=sys.stderr)
            return 1
        agents_dir = REPO_ROOT / "outputs" / "agents"
        agent_files = list(agents_dir.glob(f"{episode_id}_*.json")) if agents_dir.exists() else []
        print(f"  analyze 完成, agent 輸出: {[f.name for f in agent_files]}")

        # ----- Step 6: Layer C writeback (dry_run)；pipeline 內會自動 POST writeback -----
        print("\n[Step 6] Layer C writeback (dry_run) + POST writeback → Control Plane")
        r = subprocess.run(
            [sys.executable, "-m", "src.cli", "writeback", "--episode", str(ep_out), "--mode", "dry_run"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        if r.returncode != 0:
            print("  writeback 失敗:", r.stderr or r.stdout, file=sys.stderr)
            return 1
        writeback_path = REPO_ROOT / "outputs" / "writeback" / f"{episode_id}.json"
        stats_str = ""
        if writeback_path.exists():
            try:
                wb = json.loads(writeback_path.read_text(encoding="utf-8"))
                st = wb.get("stats", {})
                stats_str = f", stats: sightings={st.get('sightings_count', 0)}, relationships={st.get('relationships_count', 0)}, notes={st.get('notes_count', 0)}"
            except Exception:
                pass
        print(f"  apply_patch(dry_run) 完成；POST writeback → Control Plane 已由 writeback_pipeline 執行")
        print(f"  writeback 檔案: {writeback_path.relative_to(REPO_ROOT)}{stats_str}")

        # ----- Step 7: 收尾 -----
        print("\n[Step 7] 收尾")
        print("  Writeback 已寫入 outputs/writeback/" + f"{episode_id}.json 並已送至 Control Plane；")
        print("  正式環境可設定 mode=auto 寫入 OpenCTI。")
        print("\n串通成功。")
        return 0

    finally:
        if cp_proc is not None and cp_proc.poll() is None:
            cp_proc.terminate()
            try:
                cp_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cp_proc.kill()


if __name__ == "__main__":
    sys.exit(main())
