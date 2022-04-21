# -*- coding: utf-8 -*-
from docx import Document


def extract_full_text(filename: str):
    """Return the text from a document object"""
    doc = Document(filename)
    full_text = [paragraph.text for paragraph in doc.paragraphs]
    return "\n".join(full_text)
