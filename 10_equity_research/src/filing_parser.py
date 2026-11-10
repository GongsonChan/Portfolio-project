import re
import zipfile
from io import BytesIO

from lxml import etree

# DART's periodic-report chapters are top-level roman-numeral TITLEs ("I. 회사의 개요",
# "II. 사업의 내용", "III. 재무에 관한 사항", ...); subsections within a chapter use other
# numbering ("1.", "가.", ...). So: start collecting at the TITLE matching 사업의 내용, keep
# collecting through any non-roman-numeral subsection titles, and stop only at the NEXT
# roman-numeral-prefixed TITLE (a new top-level chapter) -- stopping at ANY title would cut the
# section off after its first subsection.
_SECTION_START_RE = re.compile(r"^[IVXivx]+\.\s*사업의\s*내용")
_ROMAN_PREFIX_RE = re.compile(r"^[IVXivx]+\.\s")

# Two different real DART filings failed with plain xml.etree.ElementTree (not well-formed):
# raw un-escaped '&' (e.g. "R&D"), and decorative brackets from the original Korean text like
# "< TV 사업부문 >" or "<ACI 코리아>" that read as broken tags. Trying to regex-patch each new
# malformation found this way doesn't scale -- lxml's recovering parser (libxml2's HTML-style
# error recovery) tolerates this entire class of real-world malformed markup directly, which is
# the standard tool for "parse real-world broken XML" rather than reinventing one.
_PARSER = etree.XMLParser(recover=True)


def extract_business_description(raw_zip: bytes, rcept_no: str, max_chars: int = 8000) -> str | None:
    """Best-effort extraction of the "사업의 내용" chapter's text from a DART document.xml ZIP.
    DART filings are HWP/PDF-converted and often not strictly well-formed, so this is a
    heuristic, not a guaranteed parse -- returns None (not a guess) when the document can't be
    recovered at all or the section header isn't found, rather than returning wrong text
    silently.

    The ZIP holds the main report PLUS attachments (감사보고서/연결감사보고서, etc.) -- the main
    document is named exactly "{rcept_no}.xml" with no suffix, matched explicitly rather than
    assuming zf.namelist()[0] (file ordering isn't guaranteed)."""
    try:
        with zipfile.ZipFile(BytesIO(raw_zip)) as zf:
            target = f"{rcept_no}.xml"
            if target not in zf.namelist():
                return None
            raw_bytes = zf.read(target)
    except zipfile.BadZipFile:
        return None

    root = etree.fromstring(raw_bytes, parser=_PARSER)
    if root is None:
        return None

    # Linear scan in document order. For TITLE elements, use the full nested text (itertext) to
    # match against the section boundaries. For every other element, use ONLY its own direct
    # .text (not itertext) -- root.iter() already visits every descendant individually, so using
    # itertext() here would collect each nested span's text twice (once via its ancestor, once
    # when iter() reaches it directly).
    collecting = False
    collected: list[str] = []
    total_len = 0
    for el in root.iter():
        is_title = _is_title(el)
        text = "".join(el.itertext()).strip() if is_title else (el.text or "").strip()
        if is_title:
            if _SECTION_START_RE.match(text):
                collecting = True
                continue
            if collecting and _ROMAN_PREFIX_RE.match(text):
                break  # next top-level chapter -- section is over
        if collecting and text:
            collected.append(text)
            total_len += len(text)
            if total_len >= max_chars:
                break

    if not collected:
        return None
    return "\n".join(collected)[:max_chars]


def _is_title(el) -> bool:
    return isinstance(el.tag, str) and el.tag.upper() == "TITLE"
