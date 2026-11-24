# GPT-SoVITS 原子节点

将本机 GPT-SoVITS 推理接入 ComfyUI 0.45.2。整包解压到 `F:\Ai\Comfy-Desktop\ComfyUI\ComfyUI\custom_nodes`，合并同名插件目录后重启 ComfyUI。节点在 `audio/GPT-SoVITS` 菜单，基础模板在「工作流 → 浏览模板」。

- `.ckpt` 和 `.pth` 权重可放在 ComfyUI 默认模型目录的 `gpt-sovits/` 子目录，或放在 ComfyUI 登记为 `gpt-sovits`（也兼容 `gpt_sovits`）的额外模型目录。支持子目录；相同相对路径优先使用先登记的目录（先查 `gpt-sovits`，再查 `gpt_sovits`，最后查本地默认目录）。BERT、HuBERT 等配套资源仍从 GPT-SoVITS 本机安装读取。
- 用 ComfyUI 的 Load Audio 读取一段 3–10 秒的参考录音，再接入「参考语音」。录音原台词是否必填由加载的 SoVITS 模型决定。将「加载模型 → 参考台词要求」接到 Preview as Text，运行后可查看结果。必填时转写主参考录音实际念的内容，并选择参考语言。
- 在待合成文本节点输入要念的新台词。合成后的 `AUDIO` 可接 Preview Audio 或 Save Audio。
- 「切句预览」或「切句结果」的 `STRING` 输出可直接接到「语音合成 → 文本」。合成按非空行分段、不重复切句，段间停顿照常生效；在合成节点用「字符串语言」选择朗读语言（默认 `all_zh`）。原来的「待合成文本」连线仍使用该文本节点的语言与切句设置。需要逐段换语气时，用各自的文本、参考语音和合成节点，最后按序拼接。
- 高级设置可调采样、分桶和并行参数。采样步数适用于带声码器的模型，CFG 适用于 v5，超采样适用于 v3。加速选项取决于本机运行环境。语气通过参考录音引导。
- 「UVR5 人声与背景分离」独立运行：把 Load Audio 接到「混合音频」，两路输出可分别接 Preview Audio 或 Save Audio。HP2 保留全部人声，HP5 尽量只保留主唱。权重依次查找 ComfyUI 登记的 `uvr5` 模型目录、默认模型目录的 `uvr5` 子目录、GPT-SoVITS 自带的 UVR5 权重目录。用人声作参考时，仍需截取 3–10 秒并按所选 SoVITS 模型要求填写真实台词。

默认运行环境为 ComfyUI PyTorch。节点自动启动推理进程，并使用 ComfyUI 的 Python 和 PyTorch。插件私有依赖放在 `private_packages`，由推理进程加载。运行环境也可手工切换成 GPT-SoVITS 自带环境。空闲 20 秒后自动释放推理进程和显存。
