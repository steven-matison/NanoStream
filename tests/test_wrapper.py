import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMD = os.path.join(ROOT, "bin", "nano-stream")


def run(env, *args):
    return subprocess.run([CMD, *args], env=env, capture_output=True, text=True, timeout=30)


def fake_env(tmp_path):
    py = tmp_path / "fake-python"
    py.write_text("#!/bin/sh\nexec sleep 300\n")       # stands in for the keeper and the renderer
    py.chmod(0o755)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return {**os.environ, "NANOSTREAM_PY": str(py), "NANOSTREAM_RUN": str(run_dir)}, run_dir


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as f:               # a zombie is gone for our purposes
        return f.read().split()[2] != "Z"


def children(pid):
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout
    return [int(p) for p in out.split()]


def test_usage(tmp_path):
    env, _ = fake_env(tmp_path)
    r = run(env)
    assert r.returncode == 64 and "keeper-restart" in r.stdout
    assert run(env, "stop").stdout.strip() == "not running"
    assert run(env, "relay-stop").stdout.strip() == "relay not running"


def test_keeper_stop_takes_the_whole_tree_by_pid(tmp_path):
    env, run_dir = fake_env(tmp_path)
    assert "keeper started" in run(env, "keeper").stdout
    loop = int((run_dir / "nano-keeper.pid").read_text())
    for _ in range(50):
        kids = children(loop)
        if kids:
            break
        time.sleep(0.1)
    assert kids
    # a bystander whose command line names the keeper script survives (no pgrep -f)
    bystander = subprocess.Popen(["sh", "-c", "exec sleep 300 # nano_keeper.py --url"])
    try:
        assert run(env, "keeper-stop").stdout.strip() == "keeper stopped"
        time.sleep(0.5)
        assert not alive(loop) and not any(alive(k) for k in kids)
        assert alive(bystander.pid)
        assert not (run_dir / "nano-keeper.pid").exists()
    finally:
        bystander.kill()


def test_mic_swap_keeps_the_other_args(tmp_path):
    env, run_dir = fake_env(tmp_path)
    run({**env, "KEEPER_ARGS": "--width 1920 --audio plughw:2,0"}, "keeper")
    try:
        r = run(env, "mic", "plughw:3,0")
        assert "keeper restarted" in r.stdout
        args = (run_dir / "nano-keeper.args").read_text().split()
        assert args.count("--audio") == 1 and "plughw:3,0" in args and "1920" in args
        run(env, "mic", "none")
        assert "--audio" not in (run_dir / "nano-keeper.args").read_text()
    finally:
        run(env, "keeper-stop")


def test_agent_pin_says_once_that_it_cannot_lower_the_nice(tmp_path):
    if os.geteuid() == 0:
        return                                           # root can lower it: nothing to say
    env, _ = fake_env(tmp_path)
    agent = subprocess.Popen(["nice", "-n", "15", "sleep", "60"])
    try:
        time.sleep(0.3)
        core = str(sorted(os.sched_getaffinity(0))[0])
        r = run({**env, "MINIFI_AGENT_PID": str(agent.pid)}, "agent-pin", core, "0")
        assert r.returncode == 1 and r.stderr == ""
        assert r.stdout.count("\n") == 1 and "nice left at 15" in r.stdout
        assert os.sched_getaffinity(agent.pid) == {int(core)}
        r = run({**env, "MINIFI_AGENT_PID": str(agent.pid)}, "agent-pin", core, "19")
        assert r.returncode == 0 and "at nice 19" in r.stdout
    finally:
        agent.kill()


def test_end_of_list_stops_and_next_reads_the_logged_name(tmp_path):
    import http.server
    import threading

    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *_):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Health)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    env, run_dir = fake_env(tmp_path)
    env["HAILO_INFER_URL"] = f"http://127.0.0.1:{srv.server_address[1]}"
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "250106_153556_00001 A.MP4").write_bytes(b"v")
    env["NANOSTREAM_CACHE"] = str(cache)
    py = tmp_path / "fake-python"                       # a renderer that plays one clip and ends the list (rc 0)
    py.write_text('#!/bin/sh\nL=/dev/null; p=""\nfor a in "$@"; do [ "$p" = "--log" ] && L=$a; p=$a; done\n'
                  'echo "playing 250106_153556_00001%20A.MP4 @ 30.00 fps" >> "$L"\nexit 0\n')
    try:
        assert "started" in run(env, "start", "/src", "rtmp://x/y").stdout
        for _ in range(100):
            if not (run_dir / "nano-stream.pid").exists():
                break
            time.sleep(0.1)
        assert not (run_dir / "nano-stream.pid").exists()         # the end of the list is not a crash: no restart
        assert "finished the list" in (run_dir / "nano-stream.log").read_text()
        r = run(env, "next")                                        # the logged name is url-encoded and upper-case
        assert "skipped 250106_153556_00001 A.MP4" in r.stdout
        assert not (cache / "250106_153556_00001 A.MP4").exists()
        assert "--start-at 250106_153556_00001 A.MP4~" in (run_dir / "nano-stream.args").read_text()
        assert (run_dir / "nano-stream.url").read_text() == "rtmp://x/y"
    finally:
        run(env, "stop")
        srv.shutdown()
