from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from . import ParsingException


@dataclass(frozen=True)
class Fragment:
    "A fragment of text within an entity annotation. Defined by starting and ending character positions"
    start: int
    end: int

    def __init__(self, start: int, end: int) -> None:
        object.__setattr__(self, "start", int(start))
        object.__setattr__(self, "end", int(end))

    def __str__(self) -> str:
        return f"{self.start} {self.end}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Fragment):
            return (self.start == other.start) and (self.end == other.end)
        else:
            return False

    def __hash__(self):
        return hash((self.start, self.end))

    def __le__(self, other: Any) -> bool:
        if isinstance(other, Fragment):
            return (self.start, self.end) <= (other.start, other.end)
        if isinstance(other, int) or isinstance(other, float):
            return (self.start <= other) and (self.end <= other)
        raise NotImplementedError

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Fragment):
            return (self.start, self.end) < (other.start, other.end)
        if isinstance(other, int) or isinstance(other, float):
            return (self.start < other) and (self.end < other)
        raise NotImplementedError


@dataclass(frozen=True)
class Annotation:
    """
    A generic type of annotation. Can be EntityAnnotation, RelationAnnotation, AttributeAnnotation, NormalizationAnnotation, NoteAnnotation, EquivalenceAnnotation. Defined by its id
    """

    id: str
    label: str

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Annotation):
            return self.id == other.id and self.label == other.label
        else:
            return False

    def __hash__(self):
        return hash((self.id, self.label))

    def set_label(self, new_label: str) -> None:
        object.__setattr__(self, "label", new_label)


@dataclass(frozen=True)
class EntityAnnotation(Annotation):
    """A type of Annotation, annotation of a text segment. Defined by a list of fragments (usually 1), the text content, and the label (category), as in ann file. e.g T1\tName 34 55\tPère Noël"""

    fragments: list[Fragment]
    content: str

    def __init__(self, id: str, label: str, fragments: list[Fragment], content: str):
        # check if id starts with T
        pattern = re.compile(r"^T\d+$")
        if pattern.match(id):
            object.__setattr__(self, "id", id)
        elif id.isdigit():
            id = "T" + id
            object.__setattr__(self, "id", id)
        else:
            raise ParsingException(
                "Badly initialized EntityAnnotation (the id should start with T, or it must exclusively contains digits)\n"
            )
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "fragments", fragments)

    @classmethod
    def from_line(cls, line: str) -> EntityAnnotation:
        if not re.match(r"^T\d+\t[\w\_-]+ \d+ \d+(;\d+ \d+)*\t.*", line):
            msg = f"Badly formed annotation (An entity with a space ?)\n{line}"
            raise ParsingException(msg)
        items = line.split("\t")
        subitems = items[1].split(" ", 1)
        fragments = [
            Fragment(int(s.split(" ")[0]), int(s.split(" ")[1]))
            for s in subitems[1].split(";")
        ]
        # content = items[2].strip("\n").strip(" ")
        content = items[2]
        return cls(items[0], subitems[0], fragments, content)

    def set_fragments(self, other: list[Fragment]):
        object.__setattr__(self, "fragments", other)

    def get_start(self) -> int:
        return self.fragments[0].start

    def get_end(self) -> int:
        return self.fragments[-1].end

    def __str__(self) -> str:
        return f"{self.id}\t{self.label} {';'.join([str(s) for s in self.fragments])}\t{self.content}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, EntityAnnotation):
            return (
                self.id == other.id
                and self.fragments == other.fragments
                and self.label == other.label
                and self.content == other.content
            )
        else:
            return False

    def __hash__(self):
        # list being immutable, list of Fragment objects cannot be hashed, so we convert it to tuple beforehand
        fragments_hash = hash(tuple(hash(f) for f in self.fragments))
        return hash((self.id, fragments_hash, self.label, self.content))

    def __le__(self, other: Any) -> bool:
        if isinstance(other, EntityAnnotation):
            if len(self.fragments) == 1 and len(other.fragments) == 1:
                return (self.get_start(), self.get_end()) <= (
                    other.get_start(),
                    other.get_end(),
                )
                # If 1st list has n fragments, the 2nd has m fragments
                # only min(m,n) fragments are compared.
            return all(
                (fr.start, fr.end) <= (o.start, o.end)
                for fr, o in zip(self.fragments, other.fragments)
            )
        if isinstance(other, Annotation):
            return self.id <= other.id
        if isinstance(other, float) or isinstance(other, int):
            return self.get_start() <= other
        raise NotImplementedError

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, EntityAnnotation):
            if len(self.fragments) == 1 and len(other.fragments) == 1:
                return (self.get_start(), self.get_end()) < (
                    other.get_start(),
                    other.get_end(),
                )
            return all(
                (fr.start, fr.end) < (o.start, o.end)
                for fr, o in zip(self.fragments, other.fragments)
            )
        if isinstance(other, Annotation):
            return self.id < other.id
        if isinstance(other, float) or isinstance(other, int):
            return self.get_start() < other
        raise NotImplementedError


@dataclass(frozen=True)
class RelationAnnotation(Annotation):
    """A type of Annotation, a relation between two EntityAnnotations."""

    # tuple str (Optional), EntityAnnotation because Relation follows these formats:
    # - R1  Negation str1:T1 str2:T2
    # - R1  Negation T1 T2
    argument1: tuple[str, EntityAnnotation]
    argument2: tuple[str, EntityAnnotation]

    def __init__(
        self,
        id: str,
        label: str,
        argument1: tuple[str, EntityAnnotation],
        argument2: tuple[str, EntityAnnotation],
    ) -> None:
        # check if id starts with R
        pattern = re.compile(r"^R\d+$")
        if pattern.match(id):
            object.__setattr__(self, "id", id)
        elif id.isdigit():
            id = "R" + id
            object.__setattr__(self, "id", id)
        else:
            raise ParsingException(
                "Badly initialized RelationAnnotation (the id argument should start with R, or it must exclusively contains digits)\n"
            )
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "argument1", argument1)
        object.__setattr__(self, "argument2", argument2)

    @classmethod
    def from_line(
        cls, line: str, entities: dict[str, EntityAnnotation]
    ) -> RelationAnnotation:
        if not re.match(r"^R\d+\t[\w\_-]+ \w+:T\d+ \w+:T\d+", line):
            raise ParsingException(f"Badly formed relation annotation\n{line}")
        items = line.split("\t")
        subitems = items[1].split(" ")
        arg1 = subitems[1].split(":")
        if arg1[1] in entities:
            ent1 = entities[arg1[1]]
        else:
            raise ParsingException(f"The referenced entity for {arg1[0]} doesn't exist")
        arg2 = subitems[2].split(":")
        if arg2[1] in entities:
            ent2 = entities[arg2[1]]
        else:
            raise ParsingException(f"The referenced entity for {arg2[0]} doesn't exist")
        return cls(items[0], subitems[0], (arg1[0], ent1), (arg2[0], ent2))

    def __str__(self) -> str:
        arg1_str = f"{self.argument1[0]}:{self.argument1[1].id}"
        arg2_str = f"{self.argument2[0]}:{self.argument2[1].id}"
        return f"{self.id}\t{self.label} {arg1_str} {arg2_str}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RelationAnnotation):
            return (
                self.id == other.id
                and self.label == other.label
                and self.argument1 == other.argument1
                and self.argument2 == other.argument2
            )
        else:
            return False

    def __hash__(self):
        return hash((self.id, self.label, self.argument1, self.argument2))


@dataclass(frozen=True)
class EquivalenceAnnotation(Annotation):
    id: Literal["*"]
    label: Literal["Equiv"]
    entities: list[EntityAnnotation]

    def __init__(self, entities: list[EntityAnnotation]) -> None:
        object.__setattr__(self, "id", "*")
        object.__setattr__(self, "label", "Equiv")
        object.__setattr__(self, "entities", entities)

    @classmethod
    def from_line(
        cls, line: str, entities: dict[str, EntityAnnotation]
    ) -> EquivalenceAnnotation:
        if not re.match(r"^\*\t[\w\_-]+( T\d)+", line):
            raise ParsingException(f"Badly formed equivalence annotation\n{line}")
        items = line.split("\t")
        entity_refs = items[1].split(" ")[1:]
        ents = [entities[ref] for ref in entity_refs if ref in entities]
        if len(ents) < 2:
            raise ParsingException(
                "There were less than 2 entities references correctly"
            )
        elif len(ents) < len(entity_refs):
            pass  # Not all the entity referenceres could be parsed
        return cls(ents)

    def __str__(self) -> str:
        return f"*\tEquiv {' '.join([e.id for e in self.entities])}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, EquivalenceAnnotation):
            return self.entities == other.entities
        else:
            return False

    def __hash__(self):
        # list being immutable, list of Entities objects cannot be hashed, so we convert it to tuple beforehand
        entities_hash = hash(tuple(hash(e) for e in self.entities))
        return hash(entities_hash)


@dataclass(frozen=True)
class EventAnnotation(Annotation):
    event_trigger: EntityAnnotation
    args: dict[str, EntityAnnotation]

    def __init__(
        self,
        id: str,
        label: str,
        event_trigger: EntityAnnotation,
        args: dict[str, EntityAnnotation],
    ) -> None:
        # check if id starts with E
        pattern = re.compile(r"^E\d+$")
        if pattern.match(id):
            object.__setattr__(self, "id", id)
        elif id.isdigit():
            id = "E" + id
            object.__setattr__(self, "id", id)
        else:
            raise ParsingException(
                "Badly initialized EventAnnotation (the id argument should start with E, or it must exclusively contains digits)\n"
            )
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "event_trigger", event_trigger)
        object.__setattr__(self, "args", args)

    @classmethod
    def from_line(
        cls, line: str, entities: dict[str, EntityAnnotation]
    ) -> EventAnnotation:
        # if not re.match('^E\d+\t[\w\_-]+:T\d+ (\w+:\w+)+',line):
        if not re.match(r"^E\d+\t[\w\_-]+:[TE]\d+( \w+:[TE]\d+)+", line):
            raise ParsingException(f"Badly formed event annotation\n{line}")
        items = line.split("\t")
        subitems = items[1].split(" ")
        subsubitems = subitems[0].split(":")
        if subsubitems[1] in entities:
            ## Next line is an unused variable
            # event_trigger = subsubitems[1]
            pass
        else:
            raise ParsingException(
                f"Event referencing a non-existing entity/event\n{line}"
            )
        args = {
            s.split(":")[0]: entities[s.split(":")[1]]
            for s in subitems[1:]
            if s.split(":")[1] in entities
        }
        if len(args) < len(subitems[1:]):
            raise ParsingException(
                f"Some arguments reference non-existing entiites/events\n{line}\nParsed: {len(args)} out of {len(subitems[1:])}"
            )
        return cls(items[0], subsubitems[0], entities[subsubitems[1]], args)

    def __str__(self) -> str:
        args_string = " ".join([f"{k}:{self.args[k].id}" for k in self.args])
        return f"{self.id}\t{self.label}:{self.event_trigger.id} {args_string}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, EventAnnotation):
            return (
                self.id == other.id
                and self.label == other.label
                and self.event_trigger == other.event_trigger
                and self.args == other.args
            )
        else:
            return False

    def __hash__(self):
        return hash((self.id, self.label, self.event_trigger, self.args))


@dataclass(frozen=True)
class AttributeAnnotation(Annotation):
    """A type of Annotation, an attribute linked to an EntityAnnotation"""

    component: Annotation
    values: Optional[list[str]] = None

    def __init__(
        self,
        id: str,
        label: str,
        component: Annotation,
        values: Optional[list[str]] = None,
    ) -> None:
        # check if id starts with A or M
        pattern = re.compile(r"^[AM]\d+$")
        if pattern.match(id):
            object.__setattr__(self, "id", id)
        elif id.isdigit():
            id = "A" + id
            object.__setattr__(self, "id", id)
        else:
            raise ParsingException(
                "Badly initialized AttributeAnnotation (the id argument should start with A or M, or it must exclusively contains digits)\n"
            )
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "component", component)
        if values is None:
            object.__setattr__(self, "values", [])
        else:
            object.__setattr__(self, "values", values)

    @classmethod
    def from_line(
        cls, line: str, entities: dict[str, Annotation]
    ) -> AttributeAnnotation:
        if not re.match(r"^[AM]\d+\t[\w\_-]+( \w+)+", line):
            raise ParsingException(f"Badly formed attribute annotation\n{line}")
        items = line.split("\t")
        subitems = items[1].split(" ")
        if not subitems[1] in entities:
            raise ParsingException("The referenced entity does not exist")
        # object.__setattr__(self, 'id', items[0])
        subitems = items[1].split(" ")
        if len(subitems) > 2:
            return cls(items[0], subitems[0], entities[subitems[1]], subitems[2:])
        else:
            return cls(items[0], subitems[0], entities[subitems[1]], [])

    def __str__(self) -> str:
        if self.values:
            return (
                f'{self.id}\t{self.label} {self.component.id} {" ".join(self.values)}'
            )
        return f"{self.id}\t{self.label} {self.component.id}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AttributeAnnotation):
            if (
                self.id == other.id
                and self.label == other.label
                and self.component == other.component
            ):
                if self.values is None:
                    return True
                else:
                    return self.values == other.values
            else:
                return False
        else:
            return False

    def __hash__(self):
        if self.values is not None:
            if isinstance(self.values, list):
                # list being immutable, list of str objects cannot be hashed, so we convert it to tuple beforehand
                values_hash = hash(tuple(hash(v) for v in self.values))
                return hash((self.id, self.label, self.component, values_hash))
        return hash((self.id, self.label, self.component, self.values))


@dataclass(frozen=True)
class NormalizationAnnotation(Annotation):
    label: Literal["Reference"]
    component: Annotation
    external_resource: tuple[str, str]
    content: str

    def __init__(
        self,
        id: str,
        component: Annotation,
        external_resource: tuple[str, str],
        content: str,
    ) -> None:
        # check if id starts with N
        pattern = re.compile(r"^N\d+$")
        if pattern.match(id):
            object.__setattr__(self, "id", id)
        elif id.isdigit():
            id = "N" + id
            object.__setattr__(self, "id", id)
        else:
            raise ParsingException(
                "Badly initialized NormalizationAnnotation (the id argument should start with N, or it must exclusively contains digits)\n"
            )
        object.__setattr__(self, "label", "Reference")
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "external_resource", external_resource)
        object.__setattr__(self, "content", content)

    @classmethod
    def from_line(
        cls, line: str, annotations: dict[str, Annotation]
    ) -> NormalizationAnnotation:
        if not re.match(r"^N\d+\t[\w\_-]+ \w+ \w+:\w+\t.+", line):
            raise ParsingException(f"Badly formed normalization annotation\n{line}")
        items = line.split("\t")
        subitems = items[1].split(" ")
        if not subitems[1] in annotations:
            raise ParsingException("The referenced entity does not exist")
        e_resource = subitems[2].split(":")
        return cls(
            items[0], annotations[subitems[1]], (e_resource[0], e_resource[1]), items[2]
        )

    def __str__(self) -> str:
        return f"{self.id}\t{self.label} {self.component.id} {self.external_resource[0]}:{self.external_resource[1]}\t{self.content}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, NormalizationAnnotation):
            return (
                self.id == other.id
                and self.label == other.label
                and self.component == other.component
                and self.external_resource == other.external_resource
                and self.content == other.content
            )
        else:
            return False

    def __hash__(self):
        return hash(
            (self.id, self.label, self.component, self.external_resource, self.content)
        )


@dataclass(frozen=True)
class NoteAnnotation(Annotation):
    component: Optional[Union[Annotation, str]]
    value: str

    def __init__(
        self,
        id: str,
        label: str,
        value: str,
        component: Optional[Union[Annotation, str]],
    ) -> None:
        # check if id starts with #
        pattern = re.compile(r"^#\d+$")
        if pattern.match(id):
            object.__setattr__(self, "id", id)
        elif id.isdigit():
            id = "#" + id
            object.__setattr__(self, "id", id)
        else:
            raise ParsingException(
                "Badly initialized NoteAnnotation (the id argument should start with #, or it must exclusively contains digits)\n"
            )
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "value", value)

    @classmethod
    def from_line(cls, line: str, annotations: dict[str, Annotation]) -> NoteAnnotation:
        if not re.match(r"^#\d*\t[\w\_\.-]+ \w+\t.*", line):
            raise ParsingException(f"Badly formed note annotation\n{line}")
        items = line.split("\t")
        subitems = items[1].split(" ")
        if len(subitems) > 1:
            if subitems[1] in annotations:
                return cls(items[0], subitems[0], items[2], annotations[subitems[1]])
            else:
                return cls(items[0], subitems[0], items[2], subitems[1])
        else:
            return cls(items[0], subitems[0], items[2], None)

    def __str__(self) -> str:
        if type(self.component) == str:
            return f"{self.id}\t{self.label} {self.component}\t{self.value}"
        else:
            return f"{self.id}\t{self.label} {self.component.id}\t{self.value}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, NoteAnnotation):
            if (
                self.id == other.id
                and self.label == other.label
                and self.value == other.value
            ):
                if self.component is None:
                    return True
                else:
                    return self.component == other.component
            else:
                return False
        else:
            return False

    def __hash__(self):
        return hash((self.id, self.label, self.value, self.component))
