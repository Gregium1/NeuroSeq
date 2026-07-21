from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    background: str
    card: str
    card_alt: str
    text: str
    muted: str
    border: str
    navy: str
    blue: str
    blue_hover: str
    purple: str
    purple_soft: str
    teal: str
    success: str
    danger: str
    console_bg: str
    console_fg: str
    input_bg: str


LIGHT = Palette(
    background="#EEF2F8",
    card="#FFFFFF",
    card_alt="#F6F3FF",
    text="#162033",
    muted="#667085",
    border="#D7DCE5",
    navy="#173B5E",
    blue="#2F6FDB",
    blue_hover="#245CC0",
    purple="#7567D8",
    purple_soft="#EAE6FF",
    teal="#17A589",
    success="#2D9D78",
    danger="#C94C5C",
    console_bg="#FBFCFE",
    console_fg="#162033",
    input_bg="#FFFFFF",
)

DARK = Palette(
    background="#111827",
    card="#1B2434",
    card_alt="#25213A",
    text="#F3F5F7",
    muted="#AAB3C2",
    border="#344054",
    navy="#0C2740",
    blue="#5B8DEF",
    blue_hover="#78A2F6",
    purple="#9B8AFB",
    purple_soft="#342F55",
    teal="#32C4AA",
    success="#45C49A",
    danger="#FF7A8A",
    console_bg="#0F1724",
    console_fg="#E9EEF5",
    input_bg="#202A3A",
)


def get_palette(mode: str) -> Palette:
    return DARK if str(mode).lower() == "dark" else LIGHT
