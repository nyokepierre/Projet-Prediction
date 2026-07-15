from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

import streamlit as st


def apply_global_styles(css_path: Path) -> None:
    """Charge la feuille de style locale de l'application."""
    css = css_path.read_text(encoding="utf-8")
    st.markdown(css, unsafe_allow_html=True)


def render_sidebar_brand() -> None:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand__mark">+</div>
            <div>
                <p class="sidebar-brand__title">Accès aux soins</p>
                <p class="sidebar-brand__subtitle">Modèle prédictif des ménages de Kinshasa</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <section class="app-hero">
            <span class="app-hero__eyebrow">Économie de la santé · Machine Learning</span>
            <h1>{escape(title)}</h1>
            <p>{escape(subtitle)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_badges(items: Iterable[str]) -> None:
    badges = "".join(
        f'<span class="app-badge">{escape(item)}</span>' for item in items
    )
    st.markdown(
        f'<div class="badge-row">{badges}</div>',
        unsafe_allow_html=True,
    )


def render_section_header(
    number: str,
    title: str,
    description: str | None = None,
) -> None:
    description_html = (
        f'<p class="section-heading__description">{escape(description)}</p>'
        if description
        else ""
    )
    st.markdown(
        f"""
        <div class="section-heading">
            <div class="section-heading__number">{escape(number)}</div>
            <div>
                <h2 class="section-heading__title">{escape(title)}</h2>
                {description_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_banner(
    *,
    profile: str,
    level: str,
    probability: float,
    favorable: bool,
) -> None:
    modifier = "success" if favorable else "danger"
    icon = "✓" if favorable else "!"
    st.markdown(
        f"""
        <div class="result-banner result-banner--{modifier}">
            <div class="result-banner__icon">{icon}</div>
            <div>
                <p class="result-banner__title">{escape(profile)}</p>
                <p class="result-banner__subtitle">{escape(level)}</p>
            </div>
            <div class="result-banner__value">{probability:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer">
            <strong>Application de recherche</strong> — Analyse et prédiction du recours formel
            aux soins des ménages de Kinshasa.
        </div>
        """,
        unsafe_allow_html=True,
    )
