"""
trend_narrative.translations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lightweight translation catalog for narrative generation.

Each language has its own module (``en.py``, ``fr.py``, …) exporting a
``STRINGS`` dict.  To add a new language:

1. Copy an existing file (e.g. ``en.py``) to ``<code>.py``.
2. Translate every value in the ``STRINGS`` dict.
3. Add the language code to ``_REGISTRY`` below.

That's it — ``get_translations("<code>")`` will just work, and the new
language will appear in ``SUPPORTED_LANGUAGES``.
"""

from __future__ import annotations

from . import en, fr

# Maps language codes to their string catalogs.
# To add a new language, import the module above and add it here.
_REGISTRY: dict[str, dict[str, object]] = {
    "en": en.STRINGS,
    "fr": fr.STRINGS,
}

SUPPORTED_LANGUAGES = tuple(_REGISTRY.keys())


def get_translations(lang: str = "en") -> dict[str, object]:
    """Return the string catalog for the requested language.

    Parameters
    ----------
    lang : str
        ISO 639-1 language code (default ``"en"``).

    Raises
    ------
    ValueError
        If *lang* is not a supported language code.
    """
    try:
        return _REGISTRY[lang]
    except KeyError:
        raise ValueError(
            f"Unsupported language '{lang}'. "
            f"Supported: {', '.join(SUPPORTED_LANGUAGES)}"
        )
