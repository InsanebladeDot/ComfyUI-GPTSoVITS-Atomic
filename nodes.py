import hashlib
from pathlib import Path
import tempfile
import wave

import numpy as np
import torch
import torchaudio.functional
from comfy_api.latest import ComfyExtension, io
import folder_paths

from .gsv_model_paths import resolve_weight
from .runtime import COMFY, ROOT, RUNTIMES, model_directories, worker

MODEL = io.Custom("GSV_ATOMIC_MODEL")
TEXT = io.Custom("GSV_ATOMIC_TEXT")
REFERENCE = io.Custom("GSV_ATOMIC_REFERENCE")
SETTINGS = io.Custom("GSV_ATOMIC_SETTINGS")
CATEGORY = "audio/GPT-SoVITS"
LANGUAGES = ["all_zh", "zh", "en", "all_ja", "ja", "all_yue", "yue", "all_ko", "ko", "auto", "auto_yue"]
CUTS = {
    "按标点切": "cut5", "不切": "cut0", "四句一切": "cut1", "约五十字一切": "cut2",
    "按中文句号切": "cut3", "按英文句号切": "cut4", "手动按行独立合成": "manual",
}
DEFAULT_GPT = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
DEFAULT_SOVITS = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
UVR_MODELS = {"全部人声 (HP2)": "HP2_all_vocals.pth", "仅主唱 (HP5)": "HP5_only_main_vocal.pth"}
UVR_SAMPLE_RATE = 44100


def weights(suffix):
    names = set()
    for root in model_directories():
        for candidate in root.rglob(f"*{suffix}"):
            name = candidate.relative_to(root).as_posix()
            if name in names:
                continue
            try:
                resolve_weight((root,), name, suffix)
            except ValueError:
                continue
            names.add(name)
    default = DEFAULT_GPT if suffix == ".ckpt" else DEFAULT_SOVITS
    return sorted(names, key=lambda name: (name != default, name))


def wave_tensor(audio):
    x = audio["waveform"]
    if x.ndim != 3 or x.shape[0] != 1 or not x.shape[-1]:
        raise ValueError("需要单条非空音频。")
    return x.detach().cpu().float(), int(audio["sample_rate"])


def save_reference(audio, folder):
    x, rate = wave_tensor(audio)
    raw = (x[0].T.clamp(-1, 1).mul(32767).round().numpy().astype("<i2")).tobytes()
    tag = hashlib.sha256(str((x.shape[1], rate)).encode() + raw).hexdigest()
    path = folder / f"{tag}.wav"
    with wave.open(str(path), "wb") as file:
        file.setnchannels(x.shape[1])
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(raw)
    return str(path)


def uvr_weight(preset):
    try:
        registered = folder_paths.get_folder_paths("uvr5")
    except KeyError:
        registered = []
    roots = [*map(Path, registered), Path(folder_paths.models_dir) / "uvr5", ROOT / "tools" / "uvr5" / "uvr5_weights"]
    return resolve_weight(tuple(root.resolve() for root in roots), UVR_MODELS[preset], ".pth")


class LoadModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="GSVAtomicLoadModel", display_name="GPT-SoVITS 加载模型", category=CATEGORY,
            description="从 ComfyUI 的 GPT-SoVITS 模型目录读取权重。可将「参考台词要求」接到 Preview as Text 查看加载结果。",
            inputs=[io.Combo.Input("gpt", display_name="GPT 权重", options=weights(".ckpt") or ["(无权重)"]),
                    io.Combo.Input("sovits", display_name="SoVITS 权重", options=weights(".pth") or ["(无权重)"]),
                    io.Combo.Input("runtime", display_name="运行环境", options=RUNTIMES),
                    io.Combo.Input("device", display_name="设备", options=["cuda", "cpu"]),
                    io.Boolean.Input("fp16", display_name="CUDA 半精度", default=True)],
            outputs=[MODEL.Output(display_name="模型"), io.String.Output(display_name="参考台词要求")])

    @classmethod
    def fingerprint_inputs(cls, gpt, sovits, **kwargs):
        roots = model_directories()
        files = [resolve_weight(roots, gpt, ".ckpt"), resolve_weight(roots, sovits, ".pth")]
        return tuple((str(path), path.stat().st_mtime_ns) for path in files)

    @classmethod
    def execute(cls, gpt, sovits, runtime, device, fp16):
        roots = model_directories()
        resolve_weight(roots, gpt, ".ckpt")
        resolve_weight(roots, sovits, ".pth")
        model = dict(gpt=gpt, sovits=sovits, device=device, fp16=fp16)
        info = worker.call(runtime, op="load", model=model)
        requirement = "必填录音原台词" if info["requires_transcript"] else "录音原台词可留空"
        return io.NodeOutput((runtime, model), f"SoVITS {info['version']} · {requirement}")


class Text(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="GSVAtomicText", display_name="GPT-SoVITS 待合成文本", category=CATEGORY,
            inputs=[io.String.Input("text", display_name="待朗读内容", multiline=True, default="你好，欢迎使用语音合成。"),
                    io.Combo.Input("language", display_name="文本语言", options=LANGUAGES),
                    io.Combo.Input("cut", display_name="切句", options=list(CUTS))], outputs=[TEXT.Output(display_name="文本")])

    @classmethod
    def execute(cls, text, language, cut):
        if not text.strip():
            raise ValueError("请填写待合成文本。")
        return io.NodeOutput(dict(text=text.strip(), language=language, method=CUTS[cut]))


class Split(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="GSVAtomicSplit", display_name="GPT-SoVITS 切句预览", category=CATEGORY,
            description="输出原生字符串，可接到 Preview as Text 查看切句，或直接接到语音合成按行朗读。",
            inputs=[TEXT.Input("text", display_name="文本"), io.Combo.Input("runtime", options=RUNTIMES, display_name="运行环境")],
            outputs=[io.String.Output(display_name="切句结果")])

    @classmethod
    def execute(cls, text, runtime):
        cuts = worker.call(runtime, op="split", **text)
        return io.NodeOutput("\n".join(cuts))


class Reference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="GSVAtomicReference", display_name="GPT-SoVITS 参考语音", category=CATEGORY,
            description="接入 3 到 10 秒的主参考录音。录音原台词是否必填取决于加载的 SoVITS 模型。",
            inputs=[io.Audio.Input("audio", display_name="参考音频"),
                    io.String.Input("transcript", display_name="录音原台词", multiline=True, default="",
                        tooltip="填写主参考录音实际念的内容。支持无参考台词的模型可以留空。"),
                    io.Combo.Input("language", display_name="参考语言", options=LANGUAGES),
                    io.Autogrow.Input("aux", template=io.Autogrow.TemplatePrefix(io.Audio.Input("audio"), "aux_", min=0, max=100), optional=True)],
            outputs=[REFERENCE.Output(display_name="参考语音")])

    @classmethod
    def execute(cls, audio, transcript, language, aux=None):
        x, rate = wave_tensor(audio)
        if not 3 <= x.shape[-1] / rate <= 10:
            raise ValueError("主参考录音需要 3 到 10 秒。")
        return io.NodeOutput((audio, transcript.strip(), language, list((aux or {}).values())))


DEFAULTS = dict(top_k=15, top_p=1.0, temperature=1.0, repetition_penalty=1.35,
                batch_size=1, batch_threshold=0.75, split_bucket=True, parallel_infer=True,
                use_cuda_graph=False, use_flash_attention=False, sample_steps=None, cfg_rate=None, super_sampling=False)


class Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="GSVAtomicAdvanced", display_name="GPT-SoVITS 高级推理设置", category=CATEGORY,
            description="采样步数 0、CFG -1 使用当前模型默认值。超采样仅支持 v3，CFG 支持 v5。",
            inputs=[io.Int.Input("top_k", default=15, min=1, max=1024),
                    io.Float.Input("top_p", default=1.0, min=0.01, max=1.0, step=0.01),
                    io.Float.Input("temperature", display_name="温度", default=1.0, min=0.01, max=4.0, step=0.01),
                    io.Float.Input("repetition_penalty", display_name="重复惩罚", default=1.35, min=0.01, max=10, step=0.01),
                    io.Int.Input("batch_size", display_name="批大小", default=1, min=1, max=100),
                    io.Float.Input("batch_threshold", display_name="分桶阈值", default=0.75, min=0, max=1, step=0.01),
                    io.Boolean.Input("split_bucket", display_name="分桶", default=True),
                    io.Boolean.Input("parallel_infer", display_name="并行推理", default=True),
                    io.Boolean.Input("use_cuda_graph", display_name="CUDA Graph", default=False),
                    io.Boolean.Input("use_flash_attention", display_name="Flash Attention", default=False),
                    io.Int.Input("sample_steps", display_name="采样步数", default=0, min=0, max=256),
                    io.Float.Input("cfg_rate", display_name="CFG", default=-1, min=-1, max=20, step=0.01),
                    io.Boolean.Input("super_sampling", display_name="超采样", default=False)],
            outputs=[SETTINGS.Output(display_name="推理设置")])

    @classmethod
    def execute(cls, **settings):
        settings["sample_steps"] = settings["sample_steps"] or None
        settings["cfg_rate"] = None if settings["cfg_rate"] < 0 else settings["cfg_rate"]
        return io.NodeOutput(settings)


class Synthesize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="GSVAtomicSynthesize", display_name="GPT-SoVITS 语音合成", category=CATEGORY,
            inputs=[MODEL.Input("model", display_name="模型"),
                    io.MultiType.Input("text", types=[TEXT, io.String], display_name="文本",
                        tooltip="待合成文本节点的输出保留原有语言与切句设置；普通字符串按非空行独立合成。"),
                    REFERENCE.Input("reference", display_name="参考语音"),
                    io.Float.Input("speed", display_name="语速", default=1, min=0.25, max=4, step=0.01),
                    io.Float.Input("interval", display_name="段间停顿（秒）", default=0.3, min=0, max=30, step=0.01),
                    io.Int.Input("seed", display_name="种子", default=42, min=0, max=2**32 - 1, control_after_generate=True),
                    SETTINGS.Input("settings", display_name="高级设置", optional=True),
                    io.Combo.Input("string_language", display_name="字符串语言", options=LANGUAGES,
                        default="all_zh", optional=True, tooltip="仅接入普通字符串时使用。请选择新台词的语言。")],
            outputs=[io.Audio.Output(display_name="音频")])

    @classmethod
    def execute(cls, model, text, reference, speed, interval, seed, settings=None, string_language="all_zh"):
        if isinstance(text, str):
            text = text.strip()
            if not text:
                raise ValueError("待合成字符串为空。")
            text = dict(text=text, language=string_language, method="manual")
        runtime, model_spec = model
        audio, transcript, language, aux = reference
        with tempfile.TemporaryDirectory(prefix="comfy-gsv-audio-") as directory:
            folder = Path(directory)
            output = str(folder / "output.wav")
            payload = dict(DEFAULTS if settings is None else settings,
                text=text["text"], text_lang=text["language"], text_split_method=text["method"],
                ref_audio_path=save_reference(audio, folder), prompt_text=transcript, prompt_lang=language,
                aux_ref_audio_paths=[save_reference(item, folder) for item in aux],
                speed_factor=speed, fragment_interval=interval, seed=seed)
            worker.call(runtime, op="render", model=model_spec, payload=payload, output=output)
            with wave.open(output, "rb") as wav:
                data = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
                return io.NodeOutput(dict(waveform=torch.from_numpy(data)[None, None], sample_rate=wav.getframerate()))


class SeparateVocals(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="GSVAtomicSeparateVocals", display_name="UVR5 人声与背景分离", category=CATEGORY,
            description="使用原版 UVR5 将单条音频拆成两路 44.1 kHz 音频，不需要加载 TTS 模型。",
            inputs=[io.Audio.Input("audio", display_name="混合音频"),
                    io.Combo.Input("preset", display_name="分离模型", options=list(UVR_MODELS)),
                    io.Int.Input("aggressiveness", display_name="分离强度", default=10, min=0, max=20),
                    io.Combo.Input("device", display_name="设备", options=["auto", "cuda", "cpu"])],
            outputs=[io.Audio.Output(display_name="人声"), io.Audio.Output(display_name="背景／伴奏")])

    @classmethod
    def fingerprint_inputs(cls, preset, **kwargs):
        path = uvr_weight(preset)
        return str(path), path.stat().st_mtime_ns

    @classmethod
    def execute(cls, audio, preset, aggressiveness, device):
        from scipy.io import wavfile

        samples, rate = wave_tensor(audio)
        channels = samples.shape[1]
        if channels not in (1, 2):
            raise ValueError("UVR5 需要单声道或立体声音频。")
        if rate != UVR_SAMPLE_RATE:
            samples = torchaudio.functional.resample(samples, rate, UVR_SAMPLE_RATE)
        length = samples.shape[-1]
        if not torch.any(samples):
            silent = dict(waveform=samples, sample_rate=UVR_SAMPLE_RATE)
            return io.NodeOutput(silent, silent)

        with tempfile.TemporaryDirectory(prefix="comfy-uvr5-") as directory:
            folder = Path(directory)
            source = folder / "mix.wav"
            padded = torch.nn.functional.pad(samples, (0, max(0, 4800 - length)))
            wavfile.write(source, UVR_SAMPLE_RATE, padded[0].T.contiguous().numpy())
            weight = uvr_weight(preset)
            worker.call(COMFY, op="separate_vocals", source=str(source), weight=str(weight),
                        aggressiveness=aggressiveness, device=device)
            stems = []
            for prefix in ("vocal", "instrument"):
                rate, pcm = wavfile.read(folder / f"{prefix}_{source.name}_{aggressiveness}.wav")
                waveform = torch.from_numpy(pcm.T[:channels].astype(np.float32) / 32768)[None, :, :length]
                waveform = torch.nn.functional.pad(waveform, (0, length - waveform.shape[-1]))
                stems.append(dict(waveform=waveform, sample_rate=rate))
            return io.NodeOutput(*stems)


class Pause(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="GSVAtomicPause", display_name="GPT-SoVITS 片尾加停顿", category=CATEGORY,
            inputs=[io.Audio.Input("audio", display_name="音频"),
                    io.Int.Input("pause_ms", display_name="片尾停顿（毫秒）", default=300, min=0, max=30000)],
            outputs=[io.Audio.Output(display_name="音频")])

    @classmethod
    def execute(cls, audio, pause_ms):
        samples, rate = wave_tensor(audio)
        padded = torch.nn.functional.pad(samples, (0, round(pause_ms * rate / 1000)))
        return io.NodeOutput(dict(waveform=padded, sample_rate=rate))


class Join(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="GSVAtomicJoin", display_name="GPT-SoVITS 按序拼接音频", category=CATEGORY,
            inputs=[io.Autogrow.Input("clips", template=io.Autogrow.TemplatePrefix(io.Audio.Input("audio"), "clip_", min=2, max=100))],
            outputs=[io.Audio.Output(display_name="音频")])

    @classmethod
    def execute(cls, clips):
        sources = [wave_tensor(clips[key]) for key in sorted(clips, key=lambda key: int(key.split("_")[-1]))]
        rate = max(rate for _, rate in sources)
        channels = max(samples.shape[1] for samples, _ in sources)
        combined = []
        for samples, source_rate in sources:
            if samples.shape[1] == 1 and channels > 1:
                samples = samples.expand(-1, channels, -1)
            if samples.shape[1] != channels:
                raise ValueError("无法拼接声道数不同的多声道片段。")
            combined.append(torchaudio.functional.resample(samples, source_rate, rate) if source_rate != rate else samples)
        return io.NodeOutput(dict(waveform=torch.cat(combined, dim=-1), sample_rate=rate))


class AtomicExtension(ComfyExtension):
    async def get_node_list(self):
        return [LoadModel, Text, Reference, Synthesize, Split, Advanced, Pause, Join, SeparateVocals]

