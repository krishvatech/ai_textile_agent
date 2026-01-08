from __future__ import annotations

from typing import Dict, List, Optional


STYLE_PRESETS: Dict[str, str] = {
    "Minimal": "minimal, clean studio styling with soft tones",
    "Luxury": "luxury editorial styling, premium materials, refined lighting",
    "Lifestyle": "light lifestyle styling, cozy but uncluttered",
    "Street": "modern streetwear styling, energetic but clean",
    "Festival": "festive styling with vibrant but tasteful accents",
}

BACKGROUND_PRESETS: Dict[str, str] = {
    "white": "pure white studio background",
    "lightgray": "light gray studio background",
    "lifestyle": "minimal lifestyle set, subtle props, not busy",
}

POSE_SETS: Dict[str, List[str]] = {
    "ecom6": [
        "front view",
        "three-quarter view",
        "side view",
        "back view",
        "detail close-up of material and stitching",
        "flat lay shot",
    ],
    "model6": [
        "model pose 1, relaxed stance",
        "model pose 2, walking",
        "model pose 3, seated",
        "model pose 4, arms crossed",
        "model pose 5, dynamic turn",
        "model pose 6, candid",
    ],
    "angles6": [
        "front angle",
        "back angle",
        "left side angle",
        "right side angle",
        "close-up detail angle",
        "macro texture detail",
    ],
}


def _join_style_bits(style_preset: Optional[str], reference_style: Optional[Dict[str, str]]) -> str:
    parts: List[str] = []
    if style_preset:
        preset = STYLE_PRESETS.get(style_preset, "").strip()
        if preset:
            parts.append(preset)
    if reference_style:
        lighting = reference_style.get("lighting")
        background = reference_style.get("background")
        mood = reference_style.get("mood")
        camera = reference_style.get("camera")
        composition = reference_style.get("composition")
        colors = reference_style.get("colors")
        notes = reference_style.get("notes")
        if lighting:
            parts.append(f"lighting: {lighting}")
        if background:
            parts.append(f"background: {background}")
        if mood:
            parts.append(f"mood: {mood}")
        if camera:
            parts.append(f"camera: {camera}")
        if composition:
            parts.append(f"composition: {composition}")
        if colors:
            parts.append(f"colors: {colors}")
        if notes:
            parts.append(f"notes: {notes}")
    return "; ".join([p for p in parts if p])


def build_prompts(
    *,
    workflow: str,
    background: str,
    pose_set: str,
    strict_garment: bool,
    style_preset: Optional[str],
    reference_style: Optional[Dict[str, str]],
    num_images: int,
    subject_hint: Optional[str],
) -> List[str]:
    bg_text = BACKGROUND_PRESETS.get(background, background or "studio background")
    style_bits = _join_style_bits(style_preset, reference_style)

    pose_variations = POSE_SETS.get(pose_set, POSE_SETS["ecom6"])
    prompts: List[str] = []

    for i in range(num_images):
        pose = pose_variations[i % len(pose_variations)]
        parts = []
        if subject_hint:
            parts.append(subject_hint)
        if workflow == "accessory":
            parts.append("focus on the product placement and realism")
        if workflow == "clothing":
            parts.append("realistic fabric drape and fit")
        parts.append(f"{pose}")
        parts.append(bg_text)
        if style_bits:
            parts.append(style_bits)
        if strict_garment:
            parts.append("preserve exact logo, pattern, colors, and typography; no text changes")
        parts.append("high detail, product centered, professional ecommerce quality")
        prompts.append(". ".join(parts))

    return prompts
