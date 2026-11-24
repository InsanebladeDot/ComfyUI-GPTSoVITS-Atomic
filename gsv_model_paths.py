from pathlib import Path


def resolve_weight(roots, name, suffix):
    relative = Path(name)
    if relative.drive or relative.root or ".." in relative.parts or relative.suffix.lower() != suffix:
        raise ValueError(f"无效的 GPT-SoVITS 权重文件名：{name}。")
    for root in roots:
        try:
            path = (root / relative).resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            continue
        if path.is_relative_to(root) and path.suffix.lower() == suffix and path.is_file():
            return path
    directories = "；".join(map(str, roots))
    raise ValueError(f"找不到 {name}。请在 ComfyUI 登记的 GPT-SoVITS 模型目录中选择权重：{directories}")
