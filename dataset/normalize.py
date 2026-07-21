#!/usr/bin/env python3
"""
Shared normalization + dimension parsing — used by ALL scoring so that every
stage judges text equivalence the same way. Import this everywhere; do not
re-implement normalization locally.

What it unifies (so cosmetically-different strings compare equal):
  - case + whitespace          "R 10 " == "r10"
  - decimal comma              "25,5"  == "25.5"
  - diameter symbols           "⌀25" == "Ø25" == "DIA 25" == "PHI25"  -> "diam 25"
  - radius                     "R10"  == "RAD 10"                     -> "rad 10"
  - degree                     "45°"  == "45 deg"                     -> "45 deg"
  - plus/minus                 "±0.1" == "+/-0.1"                     -> "±0.1"
  - multiplication             "3x M6" == "3 × M6"                    -> "3 x m6"

Dimension parsing (`parse_dimension`) extracts the *semantic* fields that the
dimension metric scores on — nominal value, tolerance, symbol, unit — from the
raw annotated text, so "⌀25±0.1" and "DIA 25 +/- 0.10 mm" score as equal.
"""
import re

# ---- symbol unification (applied before comparison) ------------------------
_DIAM = ["⌀", "Ø", "ø", "∅"]
_SUBS = [
    (r"[，]", ","),
    (r"\s*\+/-\s*", "±"),
    (r"\s*\+-\s*", "±"),
]


def _unify_symbols(s: str) -> str:
    for d in _DIAM:
        s = s.replace(d, " diam ")
    # word forms of diameter / radius / degree
    s = re.sub(r"\bdia\b\.?", " diam ", s, flags=re.I)
    s = re.sub(r"\bphi\b", " diam ", s, flags=re.I)
    s = re.sub(r"\bdiameter\b", " diam ", s, flags=re.I)
    s = re.sub(r"\bradius\b", " rad ", s, flags=re.I)
    s = re.sub(r"\brad\b\.?", " rad ", s, flags=re.I)
    # bare radius shorthand: "R10" / "R 10" -> "rad 10"
    s = re.sub(r"\br\s*(?=\d)", " rad ", s, flags=re.I)
    s = re.sub(r"°", " deg ", s)
    s = re.sub(r"\bdegrees?\b", " deg ", s, flags=re.I)
    s = re.sub(r"[×✕]", " x ", s)
    for pat, rep in _SUBS:
        s = re.sub(pat, rep, s)
    return s


def normalize_text(s) -> str:
    """Canonical form for text-equality comparison."""
    if s is None:
        return ""
    s = str(s)
    s = _unify_symbols(s)
    s = s.replace(",", ".")            # decimal comma -> dot (after symbol pass)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def texts_equal(a, b) -> bool:
    return normalize_text(a) == normalize_text(b)


# ---- numeric equivalence ---------------------------------------------------
def to_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if v is None:
        return None
    m = re.search(r"[-+]?\d*[.,]?\d+", str(v))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "."))
    except ValueError:
        return None


def numbers_equal(a, b, rel_tol: float = 0.02, abs_tol: float = 0.5) -> bool:
    na, nb = to_number(a), to_number(b)
    if na is None or nb is None:
        return False
    return abs(na - nb) <= max(abs_tol, rel_tol * abs(nb))


# ---- dimension parsing -----------------------------------------------------
_UNIT_RE = re.compile(r"\b(mm|cm|m|in|inch|deg)\b", re.I)


def parse_dimension(text) -> dict:
    """
    Parse a dimension callout string into semantic fields. Best-effort; any
    field it can't find is None. Never raises.

    Returns: {symbol, value, tol_plus, tol_minus, unit, raw}
      symbol   : "diam" | "rad" | "deg" | None
      value    : nominal (float) or None
      tol_plus : + tolerance (float) or None
      tol_minus: - tolerance (float, stored positive) or None
      unit     : "mm" | "deg" | ... or None
    """
    raw = "" if text is None else str(text)
    norm = normalize_text(raw)

    symbol = None
    if "diam" in norm:
        symbol = "diam"
    elif "rad" in norm:
        symbol = "rad"
    elif "deg" in norm:
        symbol = "deg"

    unit_m = _UNIT_RE.search(norm)
    unit = unit_m.group(1).lower() if unit_m else ("deg" if symbol == "deg" else None)

    tol_plus = tol_minus = None
    # symmetric ±
    m = re.search(r"±\s*(\d*\.?\d+)", norm)
    if m:
        tol_plus = tol_minus = float(m.group(1))
    else:
        # asymmetric +a -b (in any order)
        mp = re.search(r"\+\s*(\d*\.?\d+)", norm)
        mm = re.search(r"-\s*(\d*\.?\d+)", norm)
        if mp:
            tol_plus = float(mp.group(1))
        if mm:
            tol_minus = float(mm.group(1))

    # nominal value = first number that isn't part of a tolerance token
    value = None
    stripped = re.sub(r"±\s*\d*\.?\d+", " ", norm)
    stripped = re.sub(r"[+\-]\s*\d*\.?\d+", " ", stripped)
    mv = re.search(r"\d*\.?\d+", stripped)
    if mv:
        value = float(mv.group(0))
    elif to_number(norm) is not None:
        value = to_number(norm)

    return {"symbol": symbol, "value": value, "tol_plus": tol_plus,
            "tol_minus": tol_minus, "unit": unit, "raw": raw}


def dims_equal(pred, gt, rel_tol: float = 0.02, abs_tol: float = 0.5) -> dict:
    """
    Field-level comparison of two dimension strings. Returns per-field booleans
    and an overall `exact` (value + tolerance + symbol all match).
    """
    p, g = parse_dimension(pred), parse_dimension(gt)
    value_ok = numbers_equal(p["value"], g["value"], rel_tol, abs_tol) if g["value"] is not None else (p["value"] is None)
    def _tol_ok(pk, gk):
        if gk is None:
            return pk is None
        return numbers_equal(pk, gk, rel_tol, abs_tol)
    tol_ok = _tol_ok(p["tol_plus"], g["tol_plus"]) and _tol_ok(p["tol_minus"], g["tol_minus"])
    symbol_ok = (p["symbol"] == g["symbol"])
    return {"value": value_ok, "tolerance": tol_ok, "symbol": symbol_ok,
            "exact": bool(value_ok and tol_ok and symbol_ok)}


if __name__ == "__main__":
    # Self-test — quick sanity checks.
    assert texts_equal("R 10", "r10")
    assert texts_equal("25,5", "25.5")
    assert texts_equal("⌀25", "DIA 25") and texts_equal("Ø25", "phi 25")
    assert texts_equal("45°", "45 deg")
    assert numbers_equal("25.0", 25) and not numbers_equal(25, 30)
    d = parse_dimension("⌀25±0.1 mm")
    assert d["symbol"] == "diam" and d["value"] == 25 and d["tol_plus"] == 0.1 and d["unit"] == "mm", d
    assert dims_equal("⌀25±0.1", "DIA 25 +/- 0.10 mm")["exact"], dims_equal("⌀25±0.1", "DIA 25 +/- 0.10 mm")
    assert not dims_equal("R25", "⌀25")["symbol"]
    print("normalize.py self-test: OK")
