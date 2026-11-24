import atexit
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import tempfile
from threading import Lock, Thread, Timer
import time

import folder_paths
from comfy.model_management import throw_exception_if_processing_interrupted

ROOT = Path(r"F:\Ai\Tool\GPT-SoVITS-v5-20261005\GPT-SoVITS-v2pro-20261006").resolve()
MODEL_FOLDERS = ("gpt-sovits", "gpt_sovits")
COMFY = "ComfyUI PyTorch"
PORTABLE = "GPT-SoVITS 自带环境"
RUNTIMES = [COMFY, PORTABLE]


def model_directories():
    paths = []
    for category in MODEL_FOLDERS:
        try:
            paths.extend(Path(path) for path in folder_paths.get_folder_paths(category))
        except KeyError:
            pass
    paths.extend(Path(folder_paths.models_dir) / category for category in MODEL_FOLDERS)
    return tuple(dict.fromkeys(path.resolve() for path in paths))


class Worker:
    def __init__(self):
        self.lock = Lock()
        self.process = None
        self.timer = None
        self.state = None
        self.runtime = None
        self.token = None

    def close(self):
        with self.lock:
            self._stop()

    def _stop(self):
        if self.timer:
            self.timer.cancel()
            self.timer = None
        self.token = None
        if self.process:
            self.process.kill()
            self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None
        if self.state:
            self.state.cleanup()
            self.state = None

    def _expire(self, token):
        with self.lock:
            if self.token is token:
                self._stop()

    def _start(self, runtime):
        self._stop()
        if runtime not in RUNTIMES:
            raise ValueError("无效的运行环境。")
        python = Path(sys.executable) if runtime == COMFY else ROOT / "runtime" / "python.exe"
        if not python.is_file() or not (ROOT / "GPT_SoVITS").is_dir():
            raise FileNotFoundError(f"找不到本地 GPT-SoVITS 源码或 Python：{python}")
        self.state = tempfile.TemporaryDirectory(prefix="comfy-gsv-worker-")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        env["PATH"] = str(ROOT / "runtime") + os.pathsep + env.get("PATH", "")
        with open(Path(self.state.name) / "worker.log", "wb") as log:
            self.process = subprocess.Popen(
                [str(python), "-I", "-B", "-u", "-X", "utf8", str(Path(__file__).with_name("worker.py")),
                 self.state.name, *map(str, model_directories())],
                cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        self.runtime = runtime
        self.replies = Queue()
        Thread(target=self._read, args=(self.process.stdout, self.replies), daemon=True).start()

    @staticmethod
    def _read(stream, replies):
        try:
            for line in stream:
                if line.startswith("GSV_REPLY "):
                    replies.put(json.loads(line.removeprefix("GSV_REPLY ")))
        finally:
            replies.put(None)

    def call(self, runtime, **request):
        with self.lock:
            if self.timer:
                self.timer.cancel()
            self.token = None
            try:
                if self.process is None or self.process.poll() is not None or self.runtime != runtime:
                    self._start(runtime)
                self.process.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
                self.process.stdin.flush()
                deadline = time.monotonic() + 900
                while True:
                    throw_exception_if_processing_interrupted()
                    if time.monotonic() > deadline:
                        raise TimeoutError("GPT-SoVITS 推理超过 15 分钟。")
                    try:
                        reply = self.replies.get(timeout=0.2)
                        break
                    except Empty:
                        continue
                if reply is None:
                    log = Path(self.state.name, "worker.log").read_bytes()[-2500:].decode("utf-8", errors="replace")
                    raise RuntimeError(f"GPT-SoVITS 进程退出。\n{log}")
                if "error" in reply:
                    raise RuntimeError(f"GPT-SoVITS（{runtime}）{reply['error']}")
            except BaseException:
                self._stop()
                raise
            self.token = object()
            self.timer = Timer(20, self._expire, (self.token,))
            self.timer.daemon = True
            self.timer.start()
            return reply["result"]


worker = Worker()
atexit.register(worker.close)

