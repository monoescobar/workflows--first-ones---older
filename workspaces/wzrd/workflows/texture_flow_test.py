"""
Test script for calling the TextureFlow endpoint on the deployed Modal ComfyUI app.

Usage:

python texture_flow_test.py --images https://storage.googleapis.com/public-assets-xander/A_workbox/init_imgs/test_01.jpg https://storage.googleapis.com/public-assets-xander/A_workbox/init_imgs/test_02.jpg --diffusion_mask https://storage.googleapis.com/public-assets-xander/A_workbox/init_imgs/mask_img.webp --no_upscale




"""

import argparse
import os
import urllib.request
from urllib.parse import urlparse

import modal

# S3/CloudFront settings for downloading outputs
BUCKET_PREFIX = os.getenv("AWS_BUCKET_NAME", "edenartlab-stage-data")
AWS_REGION = os.getenv("AWS_REGION_NAME", "us-east-1")
CLOUDFRONT_URL = os.getenv("CLOUDFRONT_URL")

# Single source of truth for all default values
DEFAULTS = {
    "n_seconds": 5.0,
    "width": 512,
    "height": 512-128,
    "base_model": "SD15/juggernaut_reborn.safetensors",
    "use_controlnet1": False,
    "control_input": None,
    "diffusion_mask": None,
    "preprocessor1": "Scribble_XDoG_Preprocessor",
    "controlnet_strength1": 0.45,
    "denoise": 1.0,
    "control_input_fit_strategy": "fill / crop",
    "map_shape_input_to_ip_masks": False,
    "mapping_mode": "concentric_circles_outwards",
    "n_steps": 6,
    "motion_scale": 1.1,
    "use_upscale": False,
    "upscale_resolution": 1280,
    "upscale_esrgan": False,
    "seed": None,
    "app_name": "comfyui-wzrd-STAGE",
    "cls_name": "ComfyUIPremium",
}


def run_texture_flow(
    images: list[str],
    n_seconds: float = DEFAULTS["n_seconds"],
    width: int = DEFAULTS["width"],
    height: int = DEFAULTS["height"],
    base_model: str = DEFAULTS["base_model"],
    use_controlnet1: bool = DEFAULTS["use_controlnet1"],
    control_input: str | None = DEFAULTS["control_input"],
    diffusion_mask: str | None = DEFAULTS["diffusion_mask"],
    preprocessor1: str = DEFAULTS["preprocessor1"],
    controlnet_strength1: float = DEFAULTS["controlnet_strength1"],
    denoise: float = DEFAULTS["denoise"],
    control_input_fit_strategy: str = DEFAULTS["control_input_fit_strategy"],
    map_shape_input_to_ip_masks: bool = DEFAULTS["map_shape_input_to_ip_masks"],
    mapping_mode: str = DEFAULTS["mapping_mode"],
    n_steps: int = DEFAULTS["n_steps"],
    motion_scale: float = DEFAULTS["motion_scale"],
    use_upscale: bool = DEFAULTS["use_upscale"],
    upscale_resolution: int = DEFAULTS["upscale_resolution"],
    upscale_esrgan: bool = DEFAULTS["upscale_esrgan"],
    seed: int | None = DEFAULTS["seed"],
    app_name: str = DEFAULTS["app_name"],
    cls_name: str = DEFAULTS["cls_name"],
):
    """
    Call the deployed TextureFlow Modal endpoint.

    Args:
        images: List of image URLs or file paths for style images (1-6).
        n_seconds: Video length in seconds (2.0-24.0).
        width: Video width in pixels (320-1280).
        height: Video height in pixels (320-1280).
        base_model: SD1.5 checkpoint to use.
        use_controlnet1: Enable controlnet shape guidance.
        control_input: Image/video path for controlnet shape input.
        preprocessor1: Controlnet preprocessor type.
        controlnet_strength1: Controlnet guidance strength (0.0-1.0).
        denoise: AI strength on top of shape input (0.1-1.0).
        control_input_fit_strategy: How to resize shape input.
        map_shape_input_to_ip_masks: Map shape input onto style regions.
        mapping_mode: Motion pattern for style image mapping.
        n_steps: Number of LCM denoising steps (4-14).
        motion_scale: Motion strength (0.7-1.4).
        use_upscale: Enable HD upscaling second pass.
        upscale_resolution: Max dimension for latent upscale (1024-1536).
        upscale_esrgan: Enable ESRGAN postprocessing.
        seed: Random seed for reproducibility (None = random).
        app_name: Name of the deployed Modal app.
        cls_name: Name of the Modal class to call.

    Returns:
        dict with the result (uploaded video URL, etc.)
    """
    args = {
        "images": images,
        "n_seconds": n_seconds,
        "width": width,
        "height": height,
        "base_model": base_model,
        "use_controlnet1": use_controlnet1,
        "preprocessor1": preprocessor1,
        "controlnet_strength1": controlnet_strength1,
        "denoise": denoise,
        "control_input_fit_strategy": control_input_fit_strategy,
        "map_shape_input_to_ip_masks": map_shape_input_to_ip_masks,
        "mapping_mode": mapping_mode,
        "n_steps": n_steps,
        "motion_scale": motion_scale,
        "use_upscale": use_upscale,
        "upscale_resolution": upscale_resolution,
        "upscale_esrgan": upscale_esrgan,
    }

    if control_input is not None:
        args["control_input"] = control_input

    if diffusion_mask is not None:
        args["diffusion_mask"] = diffusion_mask

    if seed is not None:
        args["seed"] = seed

    cls = modal.Cls.from_name(app_name, cls_name)
    instance = cls()
    result = instance.run.remote(tool_key="texture_flow", args=args)

    print(f"Result: {result}")

    # Download all output files from the result
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    urls = _extract_download_urls(result)
    for url, filename in urls:
        filepath = os.path.join(output_dir, filename)
        print(f"Downloading {url} -> {filepath}")
        urllib.request.urlretrieve(url, filepath)
        print(f"Saved: {filepath}")

    return result


def _get_file_url(filename: str) -> str:
    """Build a full download URL from an S3 filename key."""
    if CLOUDFRONT_URL:
        return f"{CLOUDFRONT_URL}/{filename}"
    return f"https://{BUCKET_PREFIX}.s3.{AWS_REGION}.amazonaws.com/{filename}"


def _extract_download_urls(obj, _key=None):
    """
    Recursively extract download URLs from a ComfyUI upload_result structure.

    The result from run() looks like:
    {
      "output": [
        {"filename": "abc123.mp4", "mediaAttributes": {...}}
      ]
    }

    Returns list of (url, local_filename) tuples.
    """
    urls = []
    if isinstance(obj, str) and obj.startswith("http"):
        # Already a full URL
        local_name = os.path.basename(urlparse(obj).path) or "output"
        urls.append((obj, local_name))
    elif isinstance(obj, dict):
        # Check if this dict is an uploaded file entry (has "filename" key from upload_media)
        if "filename" in obj and isinstance(obj["filename"], str):
            s3_key = obj["filename"]
            url = _get_file_url(s3_key)
            local_name = os.path.basename(s3_key)
            urls.append((url, local_name))
        else:
            for v in obj.values():
                urls.extend(_extract_download_urls(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            urls.extend(_extract_download_urls(item))
    return urls


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TextureFlow via Modal endpoint")
    parser.add_argument("--images", nargs="+", required=True, help="Style image URLs/paths (1-6)")
    parser.add_argument("--n_seconds", type=float, default=DEFAULTS["n_seconds"])
    parser.add_argument("--width", type=int, default=DEFAULTS["width"])
    parser.add_argument("--height", type=int, default=DEFAULTS["height"])
    parser.add_argument("--base_model", default=DEFAULTS["base_model"],
                        choices=["SD15/juggernaut_reborn.safetensors",
                                 "SD15/darkSushiMixMix_225D.safetensors",
                                 "SD15/protogenV22Anime_protogenV22.safetensors"])
    parser.add_argument("--use_controlnet1", action="store_true")
    parser.add_argument("--control_input", default=DEFAULTS["control_input"])
    parser.add_argument("--diffusion_mask", default=DEFAULTS["diffusion_mask"])
    parser.add_argument("--preprocessor1", default=DEFAULTS["preprocessor1"],
                        choices=["Scribble_XDoG_Preprocessor", "CannyEdgePreprocessor",
                                 "DepthAnythingV2Preprocessor", "AnyLineArtPreprocessor_aux",
                                 "DensePosePreprocessor", "none"])
    parser.add_argument("--controlnet_strength1", type=float, default=DEFAULTS["controlnet_strength1"])
    parser.add_argument("--denoise", type=float, default=DEFAULTS["denoise"])
    parser.add_argument("--control_input_fit_strategy", default=DEFAULTS["control_input_fit_strategy"],
                        choices=["stretch", "fill / crop", "pad"])
    parser.add_argument("--map_shape_input_to_ip_masks", action="store_true")
    parser.add_argument("--mapping_mode", default=DEFAULTS["mapping_mode"],
                        choices=["concentric_circles_inwards", "concentric_circles_outwards",
                                 "concentric_rectangles_inwards", "concentric_rectangles_outwards",
                                 "rotating_segments_clockwise", "rotating_segments_counter_clockwise",
                                 "pushing_segments_clockwise", "pushing_segments_counter_clockwise",
                                 "vertical_stripes_left", "vertical_stripes_right",
                                 "horizontal_stripes_up", "horizontal_stripes_down"])
    parser.add_argument("--n_steps", type=int, default=DEFAULTS["n_steps"])
    parser.add_argument("--motion_scale", type=float, default=DEFAULTS["motion_scale"])
    parser.add_argument("--no_upscale", action="store_true", help="Disable upscaling")
    parser.add_argument("--upscale_resolution", type=int, default=DEFAULTS["upscale_resolution"])
    parser.add_argument("--no_esrgan", action="store_true", help="Disable ESRGAN")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--app_name", default=DEFAULTS["app_name"])
    parser.add_argument("--cls_name", default=DEFAULTS["cls_name"])

    a = parser.parse_args()
    run_texture_flow(
        images=a.images,
        n_seconds=a.n_seconds,
        width=a.width,
        height=a.height,
        base_model=a.base_model,
        use_controlnet1=a.use_controlnet1,
        control_input=a.control_input,
        diffusion_mask=a.diffusion_mask,
        preprocessor1=a.preprocessor1,
        controlnet_strength1=a.controlnet_strength1,
        denoise=a.denoise,
        control_input_fit_strategy=a.control_input_fit_strategy,
        map_shape_input_to_ip_masks=a.map_shape_input_to_ip_masks,
        mapping_mode=a.mapping_mode,
        n_steps=a.n_steps,
        motion_scale=a.motion_scale,
        use_upscale=not a.no_upscale,
        upscale_resolution=a.upscale_resolution,
        upscale_esrgan=not a.no_esrgan and not a.no_upscale,
        seed=a.seed,
        app_name=a.app_name,
        cls_name=a.cls_name,
    )
