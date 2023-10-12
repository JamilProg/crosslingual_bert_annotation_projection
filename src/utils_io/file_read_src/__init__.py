import os
import glob
import io
import re
import json
from pathlib import Path
import pandas as pd

from os import scandir
from os.path import basename, isfile, splitext, split, join, sep

from utils_typings.annotations import (
    Annotation,
    EntityAnnotation,
    RelationAnnotation,
    EquivalenceAnnotation,
    EventAnnotation,
    AttributeAnnotation,
    NormalizationAnnotation,
    NoteAnnotation,
    AnnotationCollection,
    Document,
    DocumentCollection,
    ParsingException,
)


def parse_and_fix_ann_grammar(ann_content: str) -> str:
    """parse ann file to check:
    - if each line matches with one of our Annotations regex properly (raise Exception if no match !)
    - in the same time: check the appropriateness of the Fragment indices wrt the content
    - if not coherent, check if there's the \n issue and fix the ann accordingly
    - if there is no such line, raise Exception
    - returns the fixed ann file, if everything is good."""
    regex_dico = {
        "T": r"^T\d+\t[\w\_-]+ \d+ \d+(;\d+ \d+)*\t.*",  # Entity
        "R": r"^R\d+\t[\w\_-]+ \w+:T\d+ \w+:T\d+",  # Relation
        "A": r"^A\d+\t[\w\_-]+( \w+)+",  # Attribute
        "M": r"^M\d+\t[\w\_-]+( \w+)+",  # Attribute
        "#": r"^#\d*\t[\w\_-]+ \w+\t.*",  # Note
        "N": r"^N\d+\t[\w\_-]+ \w+ \w+:\w+\t.+",  # Normalization
        "*": r"^\*\t[\w\_-]+( T\d)+",  # Equivalence
        "E": r"^E\d+\t[\w\_-]+:[TE]\d+( \w+:[TE]\d+)+",  # Event
    }
    # Check if CRLF exists first (\r\n)
    if "\r\n" in ann_content:
        ann_contents = ann_content.split("\r\n")
    else:  # If not, LF by default
        ann_contents = ann_content.split("\n")
    for i, line in enumerate(ann_contents):
        if line == "":
            continue
        appropriate_regex = None
        if line[0] in regex_dico:
            appropriate_regex = regex_dico[line[0]]
            if re.match(appropriate_regex, line):
                # we got a match, check appropriateness of indices if it is Entity
                if line[0] != "T":  # Not an entity - no check
                    continue
                else:
                    # indices checking
                    items = line.split("\t")
                    content = items[2]
                    len_content = len(content)
                    all_indices = items[1].split(" ", 1)
                    fragments = [
                        (int(s.split(" ")[0]), int(s.split(" ")[1]))
                        for s in all_indices[1].split(";")
                    ]
                    expected_length = len(fragments) - 1
                    for fstart, fend in fragments:
                        expected_length += fend - fstart
                    if len_content == expected_length:
                        # all good, correct indices and content !
                        continue
                    else:
                        # not correct indices, maybe fixable?
                        final_content = content + " " + ann_contents[i + 1]
                        if len(final_content) == expected_length:
                            # fix ann content
                            ann_contents[i] = (
                                ann_contents[i] + " " + ann_contents[i + 1]
                            )
                            ann_contents[i + 1] = ""
                            continue
                        else:
                            # not fixable, raise Exception !
                            raise ParsingException(f"Badly formed annotation\n{line}")
            else:
                # not a match, raise Exception
                raise ParsingException(f"Badly formed annotation\n{line}")
        else:
            # not a match, raise Exception
            raise ParsingException(f"Badly formed annotation\n{line}")

    # Make the new fixed ann_content
    ann_contents_noempty = [item for item in ann_contents if item]
    ann_content_res = "\n".join(ann_contents_noempty)

    return ann_content_res


def read_from_file(path: str) -> str:
    """read any file (txt, ann) and returns the str"""
    with io.open(path, "r", encoding="utf_8", newline="") as fread:
        # using io.open instead of io to preserve newlines as is  (LF, or CRLF)
        text = fread.read()
    return text


def read_texts_from_folder(path: str) -> dict[str, str]:
    """returns a dictionary containing the content for each filename (texts)"""
    filenames = glob.glob(path + "/*.txt")
    return {basename(filename): read_from_file(filename) for filename in filenames}


def read_ann_files_from_folder(path: str) -> dict[str, str]:
    """returns a dictionary containing the content for each filename (annotations)"""
    filenames = glob.glob(path + "/*.ann")
    ret = {}
    i = 0
    for filename in filenames:
        i += 1
        ret[basename(filename)] = read_from_file(filename)
        print(i, "/", len(filenames), end="\r")
    print()
    return ret
    # return {basename(filename): read_from_file(filename) for filename in filenames}


def read_and_load_ann_file(
    path: str,
    no_duplicates: bool = True,
    sorting: bool = True,
    renumerotize: bool = True,
    grammar_check: bool = True,
    version: str = "0.0.1",
    comment: str = "Empty comment",
) -> AnnotationCollection:
    """Read ann file and returns the corresponding Annotation Collection"""
    # check if a correct path has been given
    if not path:
        print("Error: you should give the path of your ann file!")
        return None
    elif not isinstance(path, str):
        print("Error: the path of your ann file should be a str!")
        return None
    exists = isfile(path)
    if not exists:
        print("Error: the ann file located in:", path, "does not exists!")
        return None
    ann_str = read_from_file(path)
    if grammar_check:
        ann_str = parse_and_fix_ann_grammar(ann_str)
    return parse_ann_file(
        ann_str,
        sorting=sorting,
        no_duplicates=no_duplicates,
        renumerotize=renumerotize,
        version=version,
        comment=comment,
    )


def parse_ann_line(
    line: str, entities: dict[str, EntityAnnotation], annotations: dict[str, Annotation]
) -> Annotation:
    """Parses a line, identifies the type of annotation in the line, and returns a parsed Annotation with the corresponding class"""
    if not line:
        return None
    match line[0]:
        case "T":
            try:
                return EntityAnnotation.from_line(line)
            except ParsingException:
                print(
                    "Issue with parsing the line starting with T (EntityAnnotation). Returning None."
                )
                print("The line is:", line)
                return None
        case "A":
            try:
                return AttributeAnnotation.from_line(line, entities)
            except ParsingException:
                print(
                    "Issue with parsing the line starting with A (AttributeAnnotation). Returning None."
                )
                print("The line is:", line)
                return None
        case "M":
            try:
                return AttributeAnnotation.from_line(line, entities)
            except ParsingException:
                print(
                    "Issue with parsing the line starting with M (AttributeAnnotation). Returning None."
                )
                print("The line is:", line)
                return None
        case "R":
            try:
                return RelationAnnotation.from_line(line, entities)
            except ParsingException:
                print(
                    "Issue with parsing the line starting with R (RelationAnnotation). Returning None."
                )
                print("The line is:", line)
                return None
        case "E":
            try:
                return EventAnnotation.from_line(line, entities)
            except ParsingException:
                print(
                    "Issue with parsing the line starting with E (EventAnnotation). Returning None."
                )
                print("The line is:", line)
                return None
        case "N":
            try:
                return NormalizationAnnotation.from_line(line, annotations)
            except ParsingException:
                print(
                    "Issue with parsing the line starting with N (NormalizationAnnotation). Returning None."
                )
                print("The line is:", line)
                return None
        case "*":
            try:
                return EquivalenceAnnotation.from_line(line, entities)
            except ParsingException:
                print(
                    "Issue with parsing the line starting with * (EquivalenceAnnotation). Returning None."
                )
                print("The line is:", line)
                return None
        case "#":
            try:
                return NoteAnnotation.from_line(line, annotations)
            except ParsingException:
                print(
                    "Issue with parsing the line starting with # (NoteAnnotation). Returning None."
                )
                print("The line is:", line)
                return None
        case _:
            print("Issue with parsing one line. Returning None. The line is:", line)
            return None


def parse_ann_file(
    annstr: str,
    no_duplicates: bool = True,
    sorting: bool = True,
    renumerotize: bool = True,
    version: str = "0.0.1",
    comment: str = "Empty comment",
) -> AnnotationCollection:
    """Parses a whole annotation file. Returns a tuple containing:
    * A dictionary of annotations containing the
    * one list for each annotation type (i.e., a list for EntityAnnotations, one for RelationAnnotations, etc.)
    """
    # Keeping track of annotations during parsing
    entities_dict: dict[str, EntityAnnotation] = dict()
    annotations_dict: dict[str, Annotation] = dict()
    # Parsing annotations
    annotations: list[Annotation] = []
    for line in annstr.splitlines():
        ann = parse_ann_line(line, entities_dict, annotations_dict)
        if ann is None:
            continue
        # We keep track of all annotations that have an ID, EquivalenceAnnotation does not have one.
        if not isinstance(ann, EquivalenceAnnotation):
            annotations_dict.update({ann.id: ann})
        if isinstance(ann, EntityAnnotation):
            entities_dict.update({ann.id: ann})
        annotations.append(ann)

    # Create AnnotationCollection
    ann_collection = AnnotationCollection(
        annotations=annotations, version=version, comment=comment
    )
    if no_duplicates:
        ann_collection.remove_duplicates()
    if sorting:
        ann_collection.sort_annotations()
    if renumerotize:
        ann_collection.renum()
    return ann_collection


def read_document_collection_from_folder(
    path: str,
    no_duplicates_ann: bool = True,
    sort_ann: bool = True,
    renumerotize_ann: bool = True,
    grammar_check_ann: bool = True,
    version: str = "0.0.1",
    comment: str = "Empty comment",
) -> DocumentCollection:
    """Reads txt and ann from a folder and builds a DocumentCollection from that"""
    dico_txt = read_texts_from_folder(path)
    dico_ann = read_ann_files_from_folder(path)
    ann_filenames = list(dico_ann.keys())
    docs: list[Document] = []
    try:
        for txt_name in dico_txt.keys():
            if txt_name.endswith(".txt"):
                ann_name = ".ann".join(txt_name.rsplit(".txt", 1))
            elif txt_name.endswith(".TXT"):
                ann_name = ".ann".join(txt_name.rsplit(".TXT", 1))
            else:
                print(
                    "It should never happen, but a supposed-to-be txt file does not end with txt. The involved file is:",
                    path,
                    txt_name,
                )
                continue
            if ann_name not in ann_filenames:
                print(
                    "The file",
                    path,
                    txt_name,
                    "does not contain the corresponding ann file !",
                )
                continue
            else:
                full_txt_path = join(path, txt_name)
                ann_collect = read_and_load_ann_file(
                    join(path, ann_name),
                    sorting=sort_ann,
                    no_duplicates=no_duplicates_ann,
                    renumerotize=renumerotize_ann,
                    grammar_check=grammar_check_ann,
                )
                docs.append(
                    Document(
                        fullpath=full_txt_path, annotation_collections=[ann_collect]
                    )
                )
    except Exception as my_exception:
        print("Exception occurred:", str(my_exception))
        print("File:", str(txt_name))
        print("Returned None instead of DocumentCollection.")
        return None

    doc_coll = DocumentCollection(
        folderpath=path, version=version, comment=comment, documents=docs
    )
    return doc_coll
