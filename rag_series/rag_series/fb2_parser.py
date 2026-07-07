import os
from typing import Any, Dict, List, Tuple

from lxml import etree

from .config import CFG
from .text_utils import file_sha256, normalize_spaces


def local_name(elem: etree._Element) -> str:
    return etree.QName(elem).localname if isinstance(elem.tag, str) else ""


def elem_text(elem: etree._Element) -> str:
    return normalize_spaces("".join(elem.itertext()))


def parse_xml_file(file_path: str) -> etree._Element:
    parser = etree.XMLParser(
        recover=True,
        resolve_entities=False,
        remove_blank_text=False,
        huge_tree=True,
        encoding="utf-8",
    )
    return etree.parse(file_path, parser).getroot()


def xpath_first_text(root: etree._Element, xpath: str) -> str:
    values = root.xpath(xpath)
    for v in values:
        text = elem_text(v) if isinstance(v, etree._Element) else normalize_spaces(str(v))
        if text:
            return text
    return ""


def get_book_title(root: etree._Element, fallback_path: str) -> str:
    title = xpath_first_text(
        root,
        "//*[local-name()='description']//*[local-name()='title-info']//*[local-name()='book-title']",
    )
    return title or os.path.splitext(os.path.basename(fallback_path))[0]


def get_authors(root: etree._Element) -> List[str]:
    authors = []
    author_nodes = root.xpath("//*[local-name()='description']//*[local-name()='title-info']//*[local-name()='author']")
    for author in author_nodes:
        parts = []
        for tag in ["first-name", "middle-name", "last-name", "nickname"]:
            value = xpath_first_text(author, f"./*[local-name()='{tag}']")
            if value:
                parts.append(value)
        if parts:
            authors.append(" ".join(parts))
    return list(dict.fromkeys(authors))


def get_sequence_info(root: etree._Element) -> Dict[str, Any]:
    seq_nodes = root.xpath("//*[local-name()='description']//*[local-name()='title-info']//*[local-name()='sequence']")
    if not seq_nodes:
        return {}
    seq = seq_nodes[0]
    return {"sequence_name": seq.get("name"), "sequence_number": seq.get("number")}


def is_notes_body(body: etree._Element) -> bool:
    return (body.get("name") or "").lower() in {"notes", "comments"}


def node_direct_text_parts(node: etree._Element) -> List[str]:
    parts = []
    for child in node:
        tag = local_name(child)
        if tag in {"section", "title"}:
            continue
        if tag == "annotation" and not CFG.include_annotation:
            continue
        if tag == "epigraph" and not CFG.include_epigraphs:
            continue
        if tag in {"p", "subtitle", "text-author", "cite", "annotation", "epigraph"}:
            text = elem_text(child)
            if text:
                parts.append(text)
            continue
        if tag in {"poem", "stanza"} and CFG.include_poems:
            text = elem_text(child)
            if text:
                parts.append(text)
    return parts


def section_title(section: etree._Element) -> str:
    titles = []
    for title in section.xpath("./*[local-name()='title']"):
        text = elem_text(title)
        if text:
            titles.append(text)
    return " / ".join(titles).strip()


def parse_fb2_sections(file_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    root = parse_xml_file(file_path)
    book_info = {
        "file": os.path.basename(file_path),
        "title": get_book_title(root, file_path),
        "authors": get_authors(root),
        "sha256": file_sha256(file_path),
        **get_sequence_info(root),
    }

    bodies = root.xpath("./*[local-name()='body']") or root.xpath("//*[local-name()='body']")
    main_bodies = [b for b in bodies if not (CFG.skip_notes_body and is_notes_body(b))]
    if not main_bodies:
        main_bodies = bodies or [root]

    sections: List[Dict[str, Any]] = []
    section_counter = 0

    def walk_section(sec: etree._Element, inherited_title: str = "") -> None:
        nonlocal section_counter
        own_title = section_title(sec)
        chapter = own_title or inherited_title or "Без названия"
        parts = node_direct_text_parts(sec)
        if parts:
            section_counter += 1
            sections.append({
                "section_index": section_counter,
                "chapter": chapter,
                "text": "\n".join(parts),
            })
        for child in sec:
            if local_name(child) == "section":
                walk_section(child, chapter)

    for body in main_bodies:
        top_sections = [child for child in body if local_name(child) == "section"]
        if top_sections:
            for sec in top_sections:
                walk_section(sec)
        else:
            parts = node_direct_text_parts(body)
            if not parts:
                p_nodes = body.xpath(".//*[local-name()='p']")
                parts = [elem_text(p) for p in p_nodes if elem_text(p)]
            if parts:
                section_counter += 1
                sections.append({
                    "section_index": section_counter,
                    "chapter": "Без названия",
                    "text": "\n".join(parts),
                })

    return book_info, sections
