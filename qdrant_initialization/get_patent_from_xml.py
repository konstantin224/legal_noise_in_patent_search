
import pandas as pd

from lxml import etree
from pathlib import Path
from typing import Optional, List, Dict, Any

NS = {
    "pat": "http://www.wipo.int/standards/XMLSchema/ST96/Patent",
    "com": "http://www.wipo.int/standards/XMLSchema/ST96/Common",
}

LANG_ATTR = f"{{{NS['com']}}}languageCode"


def _all_text(elem: Optional[etree._Element]) -> Optional[str]:
    """Склеивает весь текст внутри элемента."""
    if elem is None:
        return None
    txt = " ".join(t.strip() for t in elem.itertext() if t and t.strip())
    return txt or None


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _get_texts_by_tag(parent: etree._Element, xpath: str) -> List[Dict[str, Any]]:
    """
    Возвращает список объектов вида:
      [{"lang": "ru", "text": "..."} , ...]
    с дедупликацией по (lang, text).
    """
    elems = parent.xpath(xpath, namespaces=NS)
    out: List[Dict[str, Any]] = []
    seen = set()
    for e in elems:
        lang = e.get(LANG_ATTR)
        text = _all_text(e)
        if not text:
            continue
        key = (lang, text)
        if key in seen:
            continue
        seen.add(key)
        out.append({"lang": lang, "text": text})
    return out


def _best_lang_text(texts: List[Dict[str, Any]], prefer_langs: tuple = ("ru", "en")) -> Optional[str]:
    """
    Берёт "лучший" текст: сначала предпочтительный язык, иначе первый.
    """
    if not texts:
        return None
    for lang in prefer_langs:
        for x in texts:
            if x.get("lang") == lang:
                return x.get("text")
    return texts[0].get("text")


def _extract_citations(scope: etree._Element) -> List[Dict[str, Any]]:
    """
    Извлекает цитаты уровня техники (INID 56) из ReferenceCitationBag.

    Возвращает список словарей:
    [
        {"lang": "ru", "text": "EP 1775786 A1, 2007.04.18. US 5965064 A, 1999.10.12.", "format": "free"},
        ...
    ]
    """
    citations: List[Dict[str, Any]] = []

    # Ищем все ReferenceCitationBag в текущем scope
    citation_bags = scope.xpath(".//pat:ReferenceCitationBag", namespaces=NS)

    for bag in citation_bags:
        # Free-format citations (наиболее распространённый случай)
        free_citations = bag.xpath("./pat:ReferenceCitationFreeFormat", namespaces=NS)
        for fc in free_citations:
            lang = fc.get(LANG_ATTR)
            text = _all_text(fc)
            if text:
                citations.append({
                    "lang": lang,
                    "text": text,
                    "format": "free",
                    "parsed": None  # можно добавить парсинг при необходимости
                })

        # Structured citations (если есть — более сложный формат)
        struct_citations = bag.xpath("./pat:ReferenceCitationStructured", namespaces=NS)
        for sc in struct_citations:
            # Извлекаем основные поля структурированной цитаты
            pub_number = _all_text(sc.find(".//pat:PublicationNumber", namespaces=NS))
            pub_date = _all_text(sc.find(".//com:PublicationDate", namespaces=NS))
            kind_code = _all_text(sc.find(".//com:PatentDocumentKindCode", namespaces=NS))
            office = _all_text(sc.find(".//com:IPOfficeCode", namespaces=NS))

            if pub_number:
                citations.append({
                    "lang": None,
                    "text": f"{pub_number} {kind_code or ''} {pub_date or ''}".strip(),
                    "format": "structured",
                    "parsed": {
                        "publication_number": pub_number,
                        "publication_date": pub_date,
                        "kind_code": kind_code,
                        "office": office
                    }
                })

    return citations


def _best_citation_text(citations: List[Dict[str, Any]], prefer_langs: tuple = ("ru", "en")) -> Optional[str]:
    """
    Возвращает "лучший" текст цитат: сначала предпочтительный язык, иначе первый.
    """
    if not citations:
        return None
    for lang in prefer_langs:
        for c in citations:
            if c.get("lang") == lang:
                return c.get("text")
    return citations[0].get("text") if citations else None


def extract_core_fields(scope: etree._Element) -> Dict[str, Any]:
    """
    Извлекает основные поля патента:
      - title, description, claims, abstract
      - citations (INID 56) — новый блок
    """
    titles_all = _get_texts_by_tag(scope, ".//pat:InventionTitle")
    desc_all = _get_texts_by_tag(scope, "./pat:Description | .//pat:Description")
    claims_all = _get_texts_by_tag(scope, "./pat:Claims | .//pat:Claims")
    abstracts_all = _get_texts_by_tag(scope, "./pat:Abstract | .//pat:Abstract")

    # Извлечение цитат уровня техники (INID 56)
    citations_all = _extract_citations(scope)

    return {
        # Основные поля (один текст — предпочтение языку)
        "title": _best_lang_text(titles_all),
        "description": _best_lang_text(desc_all),
        "claims": _best_lang_text(claims_all),
        "abstract": _best_lang_text(abstracts_all),

        # Цитаты уровня техники (INID 56)
        "citations": _best_citation_text(citations_all),  # один текст для удобства
        "citations_all": citations_all,  # полный список по языкам

        # Все языковые версии основных полей
        "title_all": titles_all,
        "description_all": desc_all,
        "claims_all": claims_all,
        "abstract_all": abstracts_all,
    }


def parse_main_record(root: etree._Element, source_file: str) -> Dict[str, Any]:
    # Идентификаторы основного документа
    ip_office = _all_text((root.xpath(".//com:IPOfficeCode", namespaces=NS) or [None])[0])
    pub_number = _all_text((root.xpath(".//pat:PublicationNumber", namespaces=NS) or [None])[0])
    kind_code = _all_text((root.xpath(".//com:PatentDocumentKindCode", namespaces=NS) or [None])[0])
    pub_date = _all_text((root.xpath(".//com:PublicationDate", namespaces=NS) or [None])[0])

    core = extract_core_fields(root)

    return {
        "source_file": source_file,
        "record_type": "main",
        "ip_office": ip_office,
        "publication_number": pub_number,
        "kind_code": kind_code,
        "publication_date": pub_date,
        **core,
        # 'analog_number': "not_a_number"
    }


def parse_analog_records(root: etree._Element, source_file: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    analogs = root.xpath(".//pat:DocumentAnalogBag/pat:DocumentAnalog", namespaces=NS)

    for a in analogs:
        ip_office = _all_text((a.xpath("./com:IPOfficeCode", namespaces=NS) or [None])[0])
        pub_number = _all_text((a.xpath("./pat:PublicationNumber", namespaces=NS) or [None])[0])
        kind_code = _all_text((a.xpath("./com:PatentDocumentKindCode", namespaces=NS) or [None])[0])
        pub_date = _all_text((a.xpath("./com:PublicationDate", namespaces=NS) or [None])[0])

        core = extract_core_fields(a)

        rows.append({
            "source_file": source_file,
            "record_type": "analog",
            "ip_office": ip_office,
            "publication_number": pub_number,
            "kind_code": kind_code,
            "publication_date": pub_date,
            **core,
            # 'analog_number': "not_a_number"
        })

    return rows


def build_df_from_st96(xml_path: str | Path) -> pd.DataFrame:
    parser = etree.XMLParser(
        recover=True,      # Пропускать ошибки синтаксиса/неймспейсов
        no_network=True,   # Не загружать внешние DTD/сущности (безопасность)
        huge_tree=True     # Разрешить парсинг очень больших документов
    )

    xml_path = Path(xml_path)
    root = etree.parse(str(xml_path),
                       parser
                       ).getroot()

    rows: List[Dict[str, Any]] = []
    if root is None:
        return None
    else:
        rows.append(parse_main_record(root, str(xml_path)))         # Основной документ
        
    return pd.DataFrame(rows)

