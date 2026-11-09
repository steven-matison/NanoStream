import functools
import http.server
import threading

from dashcam_walker import list_videos


def test_http_listing(tmp_path):
    (tmp_path / "b clip.mp4").write_bytes(b"1")
    (tmp_path / "a.MOV").write_bytes(b"1")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / ".hidden.mp4").write_bytes(b"1")
    (tmp_path / "day2").mkdir()
    (tmp_path / "day2" / "c.mp4").write_bytes(b"1")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}/"
        got = list_videos(base)
    finally:
        srv.shutdown()
    assert got == sorted([base + "a.MOV", base + "b%20clip.mp4", base + "day2/c.mp4"])


def test_directory_listing(tmp_path):
    (tmp_path / "x.mp4").write_bytes(b"1")
    (tmp_path / "y.jpg").write_bytes(b"1")
    assert list_videos(str(tmp_path)) == [str(tmp_path / "x.mp4")]
    assert list_videos(str(tmp_path / "x.mp4")) == [str(tmp_path / "x.mp4")]
