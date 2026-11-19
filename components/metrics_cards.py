"""Indicadores curtos, numeração brasileira e valores ausentes explícitos."""

from math import isfinite

import streamlit as st


def format_number(value: float | None, decimals: int = 1) -> str:
    """Formata valores finitos; indisponíveis nunca são convertidos em zero."""
    if decimals < 0:
        raise ValueError("A quantidade de casas decimais deve ser não negativa.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "N/D"
    if not isfinite(number):
        return "N/D"
    return f"{number:,.{decimals}f}".translate(str.maketrans({",": ".", ".": ","}))


def metric_cards(items: list[dict]) -> None:
    """Até quatro indicadores por linha, com explicações apenas no tooltip."""
    for start in range(0, len(items), 4):
        row = items[start:start + 4]
        for column, item in zip(st.columns(len(row)), row):
            with column:
                st.metric(item["label"], item["value"], help=item.get("help"), border=True)
