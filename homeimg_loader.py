"""Load trusted HomeIMG exports whose component classes live in root pipeline.py."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading


_LOAD_LOCK = threading.RLock()


def load_homeimg_pipeline(root, **kwargs):
    """Explicitly import the export code before Diffusers resolves its components.

    Like custom_pipeline + trust_remote_code=True, this executes trusted model
    code. Each export gets a distinct module; the legacy 'pipeline' alias is
    scoped to loading so sequential model loads cannot reuse another export.
    """
    root = Path(root).resolve()
    source = root / "pipeline.py"
    index = json.loads((root / "model_index.json").read_text())
    identity = str(root).encode() + source.read_bytes()
    name = "_homeimg_export_" + hashlib.sha256(identity).hexdigest()
    with _LOAD_LOCK:
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(name, None)
                raise
        module = sys.modules[name]
        previous = sys.modules.get("pipeline")
        sys.modules["pipeline"] = module
        try:
            cls = getattr(module, index["_class_name"])
            return cls.from_pretrained(str(root), **kwargs)
        finally:
            if previous is None:
                sys.modules.pop("pipeline", None)
            else:
                sys.modules["pipeline"] = previous
