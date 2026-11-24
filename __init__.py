from .nodes import AtomicExtension


async def comfy_entrypoint():
    return AtomicExtension()
