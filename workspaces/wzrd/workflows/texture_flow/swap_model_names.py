import json
import re

FILES = ["workflow.json", "workflow_api.json"]

# Simple string replacements: (old, new)
SIMPLE_SWAPS = [
    ("RealESRGAN_x2.pth", "RealESRGAN_x2plus.pth"),
    ("dreamlike-photoreal-2.0.safetensors","SD15/juggernaut_reborn.safetensors")
]

# Regex replacements: (pattern, replacement)
REGEX_SWAPS = [
    (r"control_v11f1p_sd15_(.{3,20}?)_fp16\.safetensors", r"control_v11f1p_sd15_\1.pth"),
]


def swap(text: str, fname: str) -> str:
    for old, new in SIMPLE_SWAPS:
        count = text.count(old)
        if count:
            text = text.replace(old, new)
            print(f"  [{fname}] '{old}' -> '{new}' ({count} occurrence{'s' if count != 1 else ''})")
    for pattern, replacement in REGEX_SWAPS:
        matches = re.findall(pattern, text)
        if matches:
            text = re.sub(pattern, replacement, text)
            print(f"  [{fname}] regex '{pattern}' -> '{replacement}' ({len(matches)} match{'es' if len(matches) != 1 else ''})")
    return text


def main():
    for fname in FILES:
        with open(fname, "r") as f:
            raw = f.read()

        updated = swap(raw, fname)

        if raw != updated:
            with open(fname, "w") as f:
                f.write(updated)
            print(f"Updated {fname}")
        else:
            print(f"No changes in {fname}")


if __name__ == "__main__":
    main()
