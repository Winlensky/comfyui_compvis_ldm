import os
import folder_paths

_LDM_DIR = os.path.join(folder_paths.models_dir, "ldm")
os.makedirs(_LDM_DIR, exist_ok=True)
folder_paths.add_model_folder_path("ldm", _LDM_DIR)

# Ensure .safetensors/.ckpt appear in the dropdown; table name depends on ComfyUI version
_ext_table = getattr(folder_paths, "folder_names_and_paths", None) \
    or getattr(folder_paths, "folder_names_and_extensions", None)

if _ext_table is not None and "ldm" in _ext_table:
    _ext_table["ldm"][1].update({".ckpt", ".pt", ".pth", ".bin", ".safetensors"})

WEB_DIRECTORY = "./web"

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as e:
    import traceback
    print(f"[comfyui_compvis_ldm] FATAL: {e}")
    traceback.print_exc()
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
