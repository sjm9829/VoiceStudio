import sys, json, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voice_studio.workers.protocol import (status_event, progress_event, error_event,
                                           result_event, event)

def test_event_kinds():
    assert status_event("model_loading")["kind"] == "status"
    assert progress_event(1, 5)["index"] == 1
    assert error_event("E_X", "메시지")["kind"] == "error"
    assert result_event("/tmp/a.mp3")["output_path"] == "/tmp/a.mp3"

def test_events_json_serializable():
    for ev in (status_event("s", 1, 2), progress_event(1, 2),
               error_event("C", "m", "d"), result_event("p")):
        json.dumps(ev, ensure_ascii=False)

def test_worker_main_bad_args():
    from voice_studio.workers.worker_main import main
    assert main([]) == 2

def test_worker_main_unknown_mode(tmp_path):
    from voice_studio.workers.worker_main import main
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"mode": "??"}), encoding="utf-8")
    assert main(["worker", str(job)]) == 2

def test_worker_subprocess_emits_jsonl(tmp_path):
    root = Path(__file__).resolve().parents[1]
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"mode": "??"}), encoding="utf-8")
    r = subprocess.run([sys.executable, str(root / "src" / "voice_studio" / "workers" / "worker_main.py"),
                        str(job)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert json.loads(r.stdout.strip().splitlines()[0])["kind"] == "error"
