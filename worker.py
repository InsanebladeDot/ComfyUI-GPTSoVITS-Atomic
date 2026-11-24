"""Run the local GPT-SoVITS source using the selected Python runtime."""
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
import wave
from contextlib import nullcontext
from unittest.mock import patch


protocol = sys.stdout
sys.stdout = sys.stderr
if os.environ.get("GSV_DEBUG_IMPORTS"):
    import faulthandler
    faulthandler.dump_traceback_later(30, repeat=True)
root = Path.cwd()
if len(sys.argv) < 3:
    raise SystemExit("缺少 ComfyUI 模型目录参数。请完全退出并重新打开 ComfyUI Desktop。")
state = Path(sys.argv[1])
weights_dirs = tuple(Path(path) for path in sys.argv[2:])
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsv_model_paths import resolve_weight

bundled = Path(sys.executable).resolve().parent == root / "runtime"
if (deps := Path(__file__).resolve().with_name("private_packages")).is_dir() and not bundled:
    sys.path.insert(0, str(deps))
sys.path[:0] = [str(root / "GPT_SoVITS"), str(root)]
os.environ.setdefault("NLTK_DATA", str(root / "runtime" / "nltk_data"))
os.environ.setdefault("G2PW", "0")
os.environ.setdefault("NUMBA_CACHE_DIR", str(state / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(state / "matplotlib"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# The Windows CPython 3.13 build has no official jieba_fast wheel. GPT-SoVITS
# uses the subset of its API supplied by jieba, including POS segmentation.
if not bundled and importlib.util.find_spec("jieba_fast") is None:
    import jieba
    import jieba.posseg

    sys.modules["jieba_fast"] = jieba
    sys.modules["jieba_fast.posseg"] = jieba.posseg

pipeline = None
model_key = None
reference_key = None


def load_model(model):
    global pipeline, model_key, reference_key
    from TTS_infer_pack.TTS import TTS, TTS_Config
    import torch

    gpt = resolve_weight(weights_dirs, model["gpt"], ".ckpt")
    sovits = resolve_weight(weights_dirs, model["sovits"], ".pth")
    key = (gpt, gpt.stat().st_mtime_ns, sovits, sovits.stat().st_mtime_ns, model["device"], model["fp16"])
    if key == model_key:
        return pipeline
    if pipeline is not None:
        pipeline = model_key = reference_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    device = model["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前运行环境没有可用的 CUDA。请检查显卡环境，或在加载节点选 CPU。")
    config = TTS_Config({"custom": dict(
        TTS_Config.default_configs["v2"], device=device,
        is_half=model["fp16"] and device == "cuda",
        t2s_weights_path=str(gpt), vits_weights_path=str(sovits),
    )})
    config.configs_path = str(state / "tts_infer.yaml")
    pipeline = TTS(config)
    model_key = key
    return pipeline


def split_text(text, lang, method):
    from TTS_infer_pack.TextPreprocessor import TextPreprocessor

    if method == "manual":
        return [line.strip() for line in text.splitlines() if line.strip()]
    processor = TextPreprocessor(None, None, "cpu")
    return processor.pre_seg_text(processor.replace_consecutive_punctuation(text), lang, method)


def load_pcm16(path):
    import numpy as np
    import torch

    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError("参考音频不是 16 位 PCM WAV。")
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        audio = pcm.reshape(-1, wav.getnchannels()).T.astype(np.float32) / 32768
        return torch.from_numpy(audio), wav.getframerate()


def separate_vocals(request):
    import torch

    sys.path.insert(0, str(root / "tools" / "uvr5"))
    try:
        from vr import AudioPre
    finally:
        sys.path.pop(0)

    device = request["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("UVR5 当前没有可用的 CUDA，可改用 CPU。")
    source = Path(request["source"])
    separator = AudioPre(agg=request["aggressiveness"], model_path=request["weight"], device=device,
                         is_half=device == "cuda")
    separator._path_audio_(str(source), str(source.parent), str(source.parent), "wav")


def render(request):
    global reference_key
    import numpy as np

    engine = load_model(request["model"])
    payload = request["payload"]
    output = request["output"]
    version = engine.configs.version
    if engine.configs.use_vocoder and not payload["prompt_text"]:
        raise ValueError(f"SoVITS {version}（{Path(request['model']['sovits']).name}）需要录音原台词。"
                         "请在「参考语音 → 录音原台词」填入主参考录音实际念的内容，并选对参考语言。")
    if payload["sample_steps"] is not None and not engine.configs.use_vocoder:
        raise ValueError(f"{version} 不使用采样步数，请把高级设置中的采样步数恢复为 0。")
    if payload["cfg_rate"] is not None and not version.startswith("v5"):
        raise ValueError(f"{version} 不使用 CFG，请把高级设置中的 CFG 恢复为 -1。")
    if payload["super_sampling"] and version != "v3":
        raise ValueError("超采样仅适用于 v3，请关闭高级设置中的超采样。")
    if payload["aux_ref_audio_paths"] and engine.configs.use_vocoder:
        raise ValueError(f"{version} 不使用辅助参考音频，请移除辅助参考。")
    if payload["text_lang"] not in engine.configs.languages:
        raise ValueError(f"{version} 不支持文本语言 {payload['text_lang']}。")
    if payload["prompt_text"] and payload["prompt_lang"] not in engine.configs.languages:
        raise ValueError(f"{version} 不支持参考语言 {payload['prompt_lang']}。")

    # The upstream prompt cache keys on text alone. A language change must invalidate it.
    key = (payload["prompt_text"], payload["prompt_lang"])
    if reference_key != key:
        engine.prompt_cache["prompt_text"] = None
        reference_key = key

    method = payload["text_split_method"]
    texts = split_text(payload["text"], payload["text_lang"], method)
    if not texts:
        raise ValueError("待合成文本没有可朗读的内容。")
    if method == "manual":
        jobs = [dict(payload, text=t, text_split_method="cut0", fragment_interval=0) for t in texts]
    else:
        jobs = [payload]
    clips = []
    sample_rate = None
    import torchaudio

    for job in jobs:
        # The node supplies PCM16 WAV. Decode it locally in the ComfyUI runtime
        # so the new torchaudio loader does not need the optional TorchCodec.
        decoder = nullcontext() if bundled else patch.object(torchaudio, "load", load_pcm16)
        with decoder:
            results = list(engine.run(job))
        if len(results) != 1:
            raise RuntimeError("GPT-SoVITS 没有返回一段完整的音频。")
        rate, clip = results[0]
        if clip.size == 0 or not np.any(clip):
            raise RuntimeError("GPT-SoVITS 返回了静音。请检查文本、语言、参考录音和模型搭配。")
        if sample_rate is not None and rate != sample_rate:
            raise RuntimeError("分段输出的采样率不一致。")
        sample_rate = int(rate)
        clips.append(np.asarray(clip, dtype="<i2"))
    silence = np.zeros(round(sample_rate * payload["fragment_interval"]), dtype="<i2") if len(clips) > 1 else None
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for index, clip in enumerate(clips):
            if index and silence is not None:
                wav.writeframesraw(silence.tobytes())
            wav.writeframesraw(clip.tobytes())
    return {"version": version, "sample_rate": sample_rate}


for line in sys.stdin:
    try:
        req = json.loads(line)
        if req["op"] == "load":
            engine = load_model(req["model"])
            result = {"version": engine.configs.version, "requires_transcript": engine.configs.use_vocoder}
            del engine
        elif req["op"] == "split":
            result = split_text(req["text"], req["language"], req["method"])
        elif req["op"] == "render":
            result = render(req)
        elif req["op"] == "separate_vocals":
            result = separate_vocals(req)
        else:
            raise ValueError("无效操作")
        response = {"result": result}
    except Exception as error:
        traceback.print_exc()
        response = {"error": f"{type(error).__name__}: {error}"}
    protocol.write("GSV_REPLY " + json.dumps(response, ensure_ascii=True) + "\n")
    protocol.flush()


