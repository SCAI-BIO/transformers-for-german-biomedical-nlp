# -*- coding: utf-8 -*-
# imports
import logging
import os
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import pandas as pd

# globals

# Needed for XML parsing
META = "de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData"
SENTENCE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
ZONE = "webanno.custom.Zone"
NER = "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"
TOKEN = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"

logger = logging.getLogger(__name__)


# classes


@dataclass
class Annotation:
    begin: int
    entity: str
    zone: str
    end: int = None
    text: str = None

    def add_text(self, element: ElementTree.Element) -> str:
        text = []
        wrapped = element.findall(TOKEN)
        nested = False
        if len(wrapped) > 0:
            for token in element.findall(TOKEN):
                text.append(token.text)
            self.text = " ".join(text)
        else:
            self.text = element.text
            nested = True
        if self.end is None:
            self.end = self.begin + len(self.text)
        if nested:
            return self.text

    def to_dict(self) -> Dict:
        return {
            "start": self.begin,
            "end": self.end,
            "entity": self.entity,
            "text": self.text,
            "zone": self.zone,
        }


@dataclass
class AnnotationList:
    annotations: List[Annotation] = field(default_factory=list)

    def __add__(self, ann: Annotation):
        if isinstance(ann, Annotation):
            self.annotations.append(ann)
        else:
            logger.warning("Annotations are not of the expected type.")
        return self

    def to_df(self) -> Union[None, pd.DataFrame]:
        if len(self.annotations) > 0:
            dicts = [ann.to_dict() for ann in self.annotations]
            return pd.DataFrame(dicts)
        else:
            return None


@dataclass
class Section:
    sentences: List[str]
    entities: pd.DataFrame

    def write_to_file(self, output_prefix: str):
        text_name = f"{output_prefix}_text.txt"
        ann_name = f"{output_prefix}_annotations.csv"

        with open(text_name, "w+") as f:
            f.writelines(self.sentences)

        if self.entities is not None:
            self.entities.to_csv(ann_name)


@dataclass
class SectionList:
    sections: Dict[str, Section] = field(default_factory=dict)

    def add(self, name: str, section: Section):
        self.sections[name] = section

    def write_files(self, file_name: str, output_dir: str):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        file_folder = os.path.join(output_dir, file_name)
        if not os.path.exists(file_folder):
            os.mkdir(file_folder)

        for zone, section in self.sections.items():
            prefix = os.path.join(output_dir, file_name, zone.replace("-", "_"))
            section.write_to_file(prefix)


# functions


def process_file(
    file: str = "data/annotated/108.xml", global_offset: bool = False
) -> Union[Tuple[SectionList], Tuple[SectionList, int]]:
    root = ElementTree.parse(file).getroot()
    meta = root.find(META)

    sections = SectionList()
    current_zone = None
    current_entities = AnnotationList()
    current_text = []
    counter = 0
    if not global_offset:
        offset = 0

    for sentence in meta.findall(SENTENCE):
        items = [item for item in sentence.iter()]
        i = 0
        while i < len(items):
            xml_element = items[i]
            if xml_element.tag == ZONE:
                section = xml_element.get("Zones")
                if current_zone is None and section != "EOS":
                    current_zone = f"{str(counter)}-{section}"
                    if global_offset:
                        int(xml_element.get("begin"))
                    else:
                        offset = 0
                    counter += 1
                elif current_zone is not None and section != "EOS":
                    zone_name = current_zone if current_zone is not None else "OTHER"
                    sections.add(
                        zone_name,
                        Section(
                            sentences=" ".join(current_text).strip(),  # type: ignore
                            entities=current_entities.to_df(),
                        ),
                    )
                    current_text = []
                    current_entities = AnnotationList()
                    current_zone = f"{str(counter)}-{section}"
                    if global_offset:
                        int(xml_element.get("begin"))
                    else:
                        offset = 0
                    counter += 1
                elif current_zone is not None and section == "EOS":
                    zone_name = current_zone if current_zone is not None else "OTHER"
                    wrapped = [
                        item for item in xml_element.iter() if item != xml_element
                    ]
                    if len(wrapped) > 0:
                        # gather text within zone annotation
                        i += len(wrapped)
                        for token in wrapped:
                            if token.tag == TOKEN:
                                current_text.append(token.text)
                            elif token.tag == NER:
                                if global_offset:
                                    ann_begin = int(token.get("begin"))
                                    ann_end = int(token.get("end"))
                                else:
                                    ann_begin = len(" ".join(current_text).strip()) + 1
                                    ann_end = None

                                ann = Annotation(
                                    ann_begin,
                                    token.get("value"),
                                    current_zone.split("-")[-1]
                                    if current_zone is not None
                                    else "OTHER",
                                    end=ann_end,
                                )
                                text = ann.add_text(token)
                                if text is not None:
                                    current_text.append(token.text)
                                current_entities += ann
                            else:
                                raise Exception(
                                    f"{token.tag} was not processed. Expected text. "
                                    f"{token.get('begin')}, {token.get('end')}, {token.get('value')}"
                                )

                    sections.add(
                        zone_name,
                        Section(
                            sentences=" ".join(current_text).strip(),  # type: ignore
                            entities=current_entities.to_df(),
                        ),
                    )
                    current_text = []
                    current_entities = AnnotationList()
                    current_zone = None
                elif current_zone is None and section == "EOS":
                    pass
                else:
                    logger.debug(f"{current_zone};{section}")
                    logger.error("Zone does not meet the expected criteria.")
                    raise Exception("Zone does not meet the expected criteria.")
            elif xml_element.tag == TOKEN:
                text = xml_element.text
                if xml_element.find(NER) is None:
                    current_text.append(text)
            elif xml_element.tag == NER:
                if global_offset:
                    ann_begin = int(xml_element.get("begin"))
                    ann_end = int(xml_element.get("end"))
                else:
                    ann_begin = len(" ".join(current_text).strip()) + 1
                    ann_end = None

                ann = Annotation(
                    ann_begin,
                    xml_element.get("value"),
                    current_zone.split("-")[-1]
                    if current_zone is not None
                    else "OTHER",
                    end=ann_end,
                )
                text = ann.add_text(xml_element)
                if text is not None:
                    current_text.append(xml_element.text)
                current_entities += ann

            i += 1

        if not global_offset:
            offset += 2
        if len(current_text) > 0:
            current_text[-1] += "\n"

    output_tuple = (sections,)
    if global_offset:
        output_tuple += (int(xml_element.get("end")),)
    else:
        output_tuple += (None,)

    return output_tuple
