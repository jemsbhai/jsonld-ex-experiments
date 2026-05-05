"""EN8.10 -- Multi-Format Interop Pipeline: rdflib Baseline.

Implements the same PROV-O and RDF-Star round-trips as jsonld-ex
using rdflib + manual glue code. Measures both fidelity and LoC
for a fair comparison.

This module intentionally uses rdflib directly, with no jsonld-ex
library calls, to demonstrate the manual effort required.

Scope: PROV-O + RDF-Star only (the formats rdflib handles natively).
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Optional

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import PROV, RDF, RDFS, XSD

# ── Namespace setup ─────────────────────────────────────────────────
JEX = Namespace("https://w3id.org/jsonld-ex/")

# Annotation field mapping: jsonld-ex key -> (rdflib property, XSD type)
_PROV_O_FIELD_MAP = {
    "@confidence": (JEX.confidence, XSD.double),
    "@source": (PROV.wasAttributedTo, None),  # IRI
    "@extractedAt": (PROV.generatedAtTime, XSD.dateTime),
    "@method": (PROV.wasGeneratedBy, XSD.string),
    "@humanVerified": (JEX.humanVerified, XSD.boolean),
    "@derivedFrom": (PROV.wasDerivedFrom, None),  # IRI, multi
    "@delegatedBy": (PROV.actedOnBehalfOf, None),  # IRI, multi
    "@invalidatedAt": (PROV.invalidatedAtTime, XSD.dateTime),
    "@invalidationReason": (JEX.invalidationReason, XSD.string),
}

_RDF_STAR_FIELD_MAP = {
    "@confidence": ("confidence", XSD.double),
    "@source": ("source", None),  # IRI
    "@extractedAt": ("extractedAt", XSD.dateTime),
    "@method": ("method", XSD.string),
    "@humanVerified": ("humanVerified", XSD.boolean),
    "@derivedFrom": ("derivedFrom", None),  # IRI
    "@delegatedBy": ("delegatedBy", None),  # IRI
    "@invalidatedAt": ("invalidatedAt", XSD.dateTime),
    "@invalidationReason": ("invalidationReason", XSD.string),
    "@unit": ("unit", XSD.string),
    "@measurementUncertainty": ("measurementUncertainty", XSD.double),
    "@calibratedAt": ("calibratedAt", XSD.dateTime),
    "@calibrationMethod": ("calibrationMethod", XSD.string),
    "@calibrationAuthority": ("calibrationAuthority", XSD.string),
    "@aggregationMethod": ("aggregationMethod", XSD.string),
    "@aggregationWindow": ("aggregationWindow", XSD.string),
    "@aggregationCount": ("aggregationCount", XSD.integer),
    "@mediaType": ("mediaType", XSD.string),
    "@contentUrl": ("contentUrl", None),  # IRI
    "@contentHash": ("contentHash", XSD.string),
    "@translatedFrom": ("translatedFrom", XSD.string),
    "@translationModel": ("translationModel", XSD.string),
}


# ===================================================================
# 1. PROV-O BASELINE (rdflib manual implementation)
# ===================================================================

def baseline_to_prov_o(doc: dict[str, Any]) -> Graph:
    """Convert jsonld-ex annotated doc to PROV-O using rdflib manually.

    Replicates the mapping in jsonld_ex.owl_interop.to_prov_o()
    but using raw rdflib calls instead of the library.

    Args:
        doc: A jsonld-ex annotated document.

    Returns:
        An rdflib Graph containing PROV-O triples.
    """
    g = Graph()
    g.bind("prov", PROV)
    g.bind("xsd", XSD)
    g.bind("rdfs", RDFS)
    g.bind("jex", JEX)

    subject_str = doc.get("@id", f"_:subject-{uuid.uuid4().hex[:8]}")
    if subject_str.startswith("_:"):
        subject = BNode(subject_str[2:])
    else:
        subject = URIRef(subject_str)

    # Add type triple
    doc_type = doc.get("@type", "")
    if doc_type:
        type_uri = doc_type
        if not type_uri.startswith("http"):
            type_uri = f"http://schema.org/{type_uri}"
        g.add((subject, RDF.type, URIRef(type_uri)))

    for key, value in doc.items():
        if key.startswith("@"):
            continue

        prop_uri = URIRef(key) if key.startswith("http") else URIRef(
            f"http://schema.org/{key}"
        )

        if isinstance(value, dict) and "@value" in value:
            # Annotated value -> PROV-O Entity
            entity_id = BNode(f"entity-{uuid.uuid4().hex[:8]}")
            g.add((entity_id, RDF.type, PROV.Entity))

            # Store the actual value
            raw_val = value["@value"]
            g.add((entity_id, PROV.value, _to_literal(raw_val)))

            # Link subject to entity via property
            g.add((subject, prop_uri, entity_id))

            # Map each annotation field
            for ann_key, ann_val in value.items():
                if ann_key == "@value":
                    continue
                if ann_key not in _PROV_O_FIELD_MAP:
                    # Field not in PROV-O scope -> store as jex extension
                    jex_prop = JEX[ann_key.lstrip("@")]
                    if isinstance(ann_val, list):
                        for item in ann_val:
                            g.add((entity_id, jex_prop, _to_literal_or_uri(item)))
                    else:
                        g.add((entity_id, jex_prop, _to_literal_or_uri(ann_val)))
                    continue

                prov_prop, xsd_type = _PROV_O_FIELD_MAP[ann_key]

                if ann_key == "@source":
                    agent_id = URIRef(ann_val)
                    g.add((agent_id, RDF.type, PROV.SoftwareAgent))
                    g.add((entity_id, prov_prop, agent_id))
                elif ann_key == "@method":
                    activity_id = BNode(f"activity-{uuid.uuid4().hex[:8]}")
                    g.add((activity_id, RDF.type, PROV.Activity))
                    g.add((activity_id, RDFS.label, Literal(ann_val)))
                    g.add((entity_id, prov_prop, activity_id))
                elif ann_key == "@humanVerified":
                    if ann_val:
                        person_id = BNode(f"person-{uuid.uuid4().hex[:8]}")
                        g.add((person_id, RDF.type, PROV.Person))
                        g.add((entity_id, PROV.wasAttributedTo, person_id))
                    g.add((entity_id, JEX.humanVerified, Literal(
                        ann_val, datatype=XSD.boolean
                    )))
                elif ann_key in ("@derivedFrom", "@delegatedBy"):
                    items = ann_val if isinstance(ann_val, list) else [ann_val]
                    for item in items:
                        g.add((entity_id, prov_prop, URIRef(item)))
                elif ann_key == "@confidence":
                    g.add((entity_id, prov_prop, Literal(
                        ann_val, datatype=xsd_type
                    )))
                else:
                    g.add((entity_id, prov_prop, Literal(
                        ann_val, datatype=xsd_type
                    )))

        elif isinstance(value, str):
            if value.startswith("http://") or value.startswith("https://"):
                g.add((subject, prop_uri, URIRef(value)))
            else:
                g.add((subject, prop_uri, Literal(value)))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            g.add((subject, prop_uri, _to_literal(value)))
        elif isinstance(value, bool):
            g.add((subject, prop_uri, Literal(value, datatype=XSD.boolean)))

    return g


def baseline_from_prov_o(g: Graph) -> dict[str, Any]:
    """Reconstruct a jsonld-ex document from a PROV-O rdflib Graph.

    Reverses the mapping: finds PROV-O Entities linked from the
    subject, extracts their annotations, and rebuilds annotated values.

    Args:
        g: An rdflib Graph with PROV-O triples.

    Returns:
        A jsonld-ex document (dict).
    """
    doc: dict[str, Any] = {}

    # Find the main subject (non-Entity, non-Agent, non-Activity node)
    prov_types = {PROV.Entity, PROV.SoftwareAgent, PROV.Activity, PROV.Person}
    subjects = set()
    for s, p, o in g.triples((None, RDF.type, None)):
        if o not in prov_types:
            subjects.add(s)

    if not subjects:
        # Fall back: find any subject with outgoing non-type triples
        for s, p, o in g:
            if p != RDF.type:
                subjects.add(s)
                break

    subject = subjects.pop() if subjects else None
    if subject is None:
        return doc

    # Set @id
    if isinstance(subject, URIRef):
        doc["@id"] = str(subject)
    elif isinstance(subject, BNode):
        doc["@id"] = f"_:{subject}"

    # Set @type (strip common namespace prefixes for compaction)
    for s, p, o in g.triples((subject, RDF.type, None)):
        type_str = str(o)
        doc["@type"] = _compact_uri(type_str)

    # Find all properties from subject to PROV entities
    for s, p, o in g.triples((subject, None, None)):
        if p == RDF.type:
            continue

        prop_name = _compact_uri(str(p))

        # Check if object is a PROV Entity
        is_entity = (o, RDF.type, PROV.Entity) in g
        if is_entity:
            annotated = _rebuild_annotated_value(g, o)
            doc[prop_name] = annotated
        else:
            # Plain value
            if isinstance(o, URIRef):
                doc[prop_name] = str(o)
            elif isinstance(o, Literal):
                doc[prop_name] = _literal_to_python(o)
            elif isinstance(o, BNode):
                doc[prop_name] = f"_:{o}"

    return doc


def _rebuild_annotated_value(
    g: Graph, entity: URIRef | BNode
) -> dict[str, Any]:
    """Rebuild an annotated value dict from a PROV-O Entity."""
    result: dict[str, Any] = {}

    # Get the value
    for s, p, o in g.triples((entity, PROV.value, None)):
        result["@value"] = _literal_to_python(o)

    # Get confidence
    for s, p, o in g.triples((entity, JEX.confidence, None)):
        result["@confidence"] = _literal_to_python(o)

    # Get source (wasAttributedTo SoftwareAgent)
    for s, p, o in g.triples((entity, PROV.wasAttributedTo, None)):
        if (o, RDF.type, PROV.SoftwareAgent) in g:
            result["@source"] = str(o)

    # Get extractedAt
    for s, p, o in g.triples((entity, PROV.generatedAtTime, None)):
        result["@extractedAt"] = _literal_to_datetime_str(o)

    # Get method (wasGeneratedBy Activity -> rdfs:label)
    for s, p, o in g.triples((entity, PROV.wasGeneratedBy, None)):
        for s2, p2, o2 in g.triples((o, RDFS.label, None)):
            result["@method"] = str(o2)

    # Get humanVerified
    for s, p, o in g.triples((entity, JEX.humanVerified, None)):
        result["@humanVerified"] = _literal_to_python(o)

    # Get derivedFrom
    derived_list = []
    for s, p, o in g.triples((entity, PROV.wasDerivedFrom, None)):
        derived_list.append(str(o))
    if len(derived_list) == 1:
        result["@derivedFrom"] = derived_list[0]
    elif len(derived_list) > 1:
        result["@derivedFrom"] = derived_list

    # Get delegatedBy
    delegated_list = []
    for s, p, o in g.triples((entity, PROV.actedOnBehalfOf, None)):
        delegated_list.append(str(o))
    if len(delegated_list) == 1:
        result["@delegatedBy"] = delegated_list[0]
    elif len(delegated_list) > 1:
        result["@delegatedBy"] = delegated_list

    # Get invalidatedAt
    for s, p, o in g.triples((entity, PROV.invalidatedAtTime, None)):
        result["@invalidatedAt"] = _literal_to_datetime_str(o)

    # Get invalidationReason
    for s, p, o in g.triples((entity, JEX.invalidationReason, None)):
        result["@invalidationReason"] = str(o)

    # Get any jex: extension fields not covered above
    for s, p, o in g.triples((entity, None, None)):
        if p == RDF.type or p == PROV.value:
            continue
        p_str = str(p)
        if p_str.startswith(str(JEX)):
            local_name = p_str[len(str(JEX)):]
            ann_key = f"@{local_name}"
            if ann_key not in result:
                result[ann_key] = _literal_to_python(o) if isinstance(
                    o, Literal
                ) else str(o)

    return result


# ===================================================================
# 2. RDF-STAR BASELINE (rdflib manual implementation)
# ===================================================================

def baseline_to_rdf_star_ntriples(
    doc: dict[str, Any],
    base_subject: str | None = None,
) -> str:
    """Convert jsonld-ex annotated doc to RDF-Star N-Triples manually.

    Uses string formatting (rdflib does not natively support RDF-Star
    serialization), replicating the logic in
    jsonld_ex.owl_interop.to_rdf_star_ntriples().

    Args:
        doc: A jsonld-ex annotated document.
        base_subject: IRI for the document subject.

    Returns:
        N-Triples-Star string.
    """
    lines: list[str] = []
    subject = base_subject or doc.get("@id", "_:subject")
    if not subject.startswith("_:") and not subject.startswith("<"):
        subject = f"<{subject}>"

    for key, value in doc.items():
        if key.startswith("@"):
            continue

        prop_iri = key if key.startswith("http") else f"http://schema.org/{key}"

        if isinstance(value, dict) and "@value" in value:
            raw_val = value["@value"]
            literal = _format_ntriples_literal(raw_val)

            # Base triple
            lines.append(f"{subject} <{prop_iri}> {literal} .")

            # Build embedded triple
            embedded = f"<< {subject} <{prop_iri}> {literal} >>"

            # Annotation triples
            for ann_key, ann_val in value.items():
                if ann_key == "@value":
                    continue

                local_name = ann_key.lstrip("@")
                if ann_key not in _RDF_STAR_FIELD_MAP:
                    # Unknown annotation field -> store as jex extension
                    jex_uri = f"<https://w3id.org/jsonld-ex/{local_name}>"
                    formatted = _format_ntriples_annotation(ann_val)
                    lines.append(f"{embedded} {jex_uri} {formatted} .")
                    continue

                rdf_local, xsd_type = _RDF_STAR_FIELD_MAP[ann_key]
                jex_uri = f"<https://w3id.org/jsonld-ex/{rdf_local}>"

                if isinstance(ann_val, list):
                    for item in ann_val:
                        formatted = _format_ntriples_annotation(
                            item, is_iri=(xsd_type is None)
                        )
                        lines.append(f"{embedded} {jex_uri} {formatted} .")
                else:
                    formatted = _format_ntriples_annotation(
                        ann_val, is_iri=(xsd_type is None), xsd_type=xsd_type
                    )
                    lines.append(f"{embedded} {jex_uri} {formatted} .")

        elif isinstance(value, str):
            if value.startswith("http://") or value.startswith("https://"):
                lines.append(f"{subject} <{prop_iri}> <{value}> .")
            else:
                escaped = _escape_ntriples_str(value)
                lines.append(f'{subject} <{prop_iri}> "{escaped}" .')
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            xsd_t = f"{XSD}integer" if isinstance(value, int) else f"{XSD}double"
            lines.append(f'{subject} <{prop_iri}> "{value}"^^<{xsd_t}> .')
        elif isinstance(value, bool):
            bval = "true" if value else "false"
            lines.append(f'{subject} <{prop_iri}> "{bval}"^^<{XSD}boolean> .')

    return "\n".join(lines)


def baseline_from_rdf_star_ntriples(nt_str: str) -> dict[str, Any]:
    """Parse RDF-Star N-Triples back into a jsonld-ex document.

    Manually parses the N-Triples-Star format (rdflib cannot parse
    RDF-Star N-Triples natively).

    Args:
        nt_str: N-Triples-Star string.

    Returns:
        A jsonld-ex document (dict).
    """
    doc: dict[str, Any] = {}
    subject_id: str | None = None

    # Regex for embedded triples: << <s> <p> "val"^^<type> >>
    embedded_re = re.compile(
        r'^<< (.+?) <(.+?)> (.+?) >> <(.+?)> (.+?) \.$'
    )
    # Regex for standard triples: <s> <p> <o> .  or  <s> <p> "val"^^<type> .
    standard_re = re.compile(
        r'^(<.+?>|_:\S+) <(.+?)> (.+?) \.$'
    )

    # First pass: collect base triples and annotations
    base_triples: dict[str, dict[str, Any]] = {}  # prop -> {"@value": ..., ann_key: ann_val}
    plain_triples: dict[str, Any] = {}  # prop -> value

    for line in nt_str.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Try embedded triple first
        m = embedded_re.match(line)
        if m:
            subj_str, prop, val_str, ann_prop, ann_val_str = m.groups()
            if subject_id is None:
                subject_id = _parse_nt_term(subj_str)

            if prop not in base_triples:
                base_triples[prop] = {"@value": _parse_nt_value(val_str)}

            # Parse annotation property local name
            ann_local = ann_prop.rsplit("/", 1)[-1].rstrip(">")
            ann_key = f"@{ann_local}"
            ann_python = _parse_nt_value(ann_val_str)

            # Handle multi-valued annotations
            if ann_key in base_triples[prop]:
                existing = base_triples[prop][ann_key]
                if isinstance(existing, list):
                    existing.append(ann_python)
                else:
                    base_triples[prop][ann_key] = [existing, ann_python]
            else:
                base_triples[prop][ann_key] = ann_python

            continue

        # Try standard triple
        m = standard_re.match(line)
        if m:
            subj_str, prop, val_str = m.groups()
            if subject_id is None:
                subject_id = _parse_nt_term(subj_str)

            parsed_val = _parse_nt_value(val_str)

            # Check if this prop already has an embedded version
            if prop in base_triples:
                continue  # Base triple already captured via embedded
            else:
                plain_triples[prop] = parsed_val

    # Build document
    if subject_id:
        doc["@id"] = subject_id

    for prop, annotated in base_triples.items():
        doc[_compact_uri(prop)] = annotated

    for prop, value in plain_triples.items():
        compact = _compact_uri(prop)
        if compact not in doc:
            doc[compact] = value

    return doc


# ===================================================================
# 3. COMBINED BASELINE PIPELINE
# ===================================================================

def baseline_prov_o_roundtrip(
    doc: dict[str, Any],
) -> dict[str, Any]:
    """PROV-O round-trip using rdflib baseline."""
    g = baseline_to_prov_o(doc)
    return baseline_from_prov_o(g)


def baseline_rdf_star_roundtrip(
    doc: dict[str, Any],
) -> dict[str, Any]:
    """RDF-Star round-trip using rdflib baseline."""
    subject = doc.get("@id", "http://example.org/subject")
    nt_str = baseline_to_rdf_star_ntriples(doc, base_subject=subject)
    return baseline_from_rdf_star_ntriples(nt_str)


# ===================================================================
# 4. LOC COUNTING
# ===================================================================

def count_executable_lines(filepath: str) -> int:
    """Count executable lines in a Python file.

    Excludes: blank lines, comment-only lines, docstrings, imports.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    count = 0
    in_docstring = False

    for line in lines:
        stripped = line.strip()

        # Skip blank lines
        if not stripped:
            continue

        # Track docstrings
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if in_docstring:
                in_docstring = False
                continue
            # Single-line docstring
            if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                continue
            in_docstring = True
            continue

        if in_docstring:
            continue

        # Skip comment-only lines
        if stripped.startswith("#"):
            continue

        # Skip import lines
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue

        count += 1

    return count


# ===================================================================
# 5. HELPER FUNCTIONS
# ===================================================================

def _literal_to_datetime_str(lit: Literal | URIRef | BNode) -> str:
    """Convert an rdflib dateTime Literal back to ISO string.

    rdflib normalizes 'Z' to '+00:00' internally, so we must
    convert back to preserve the original form.
    """
    if isinstance(lit, Literal):
        val = lit.toPython()
        if hasattr(val, 'isoformat'):
            iso = val.isoformat()
            # Normalize +00:00 back to Z
            if iso.endswith("+00:00"):
                iso = iso[:-6] + "Z"
            return iso
    return str(lit)


def _compact_uri(uri: str) -> str:
    """Strip common namespace prefixes for URI compaction.

    This is the manual equivalent of JSON-LD compaction that
    jsonld-ex handles automatically via @context.
    """
    prefixes = [
        "http://schema.org/",
        "https://schema.org/",
        "http://xmlns.com/foaf/0.1/",
        "http://purl.org/dc/terms/",
    ]
    for prefix in prefixes:
        if uri.startswith(prefix):
            return uri[len(prefix):]
    return uri


def _to_literal(val: Any) -> Literal:
    """Convert a Python value to an rdflib Literal with XSD type."""
    if isinstance(val, bool):
        return Literal(val, datatype=XSD.boolean)
    if isinstance(val, int):
        return Literal(val, datatype=XSD.integer)
    if isinstance(val, float):
        return Literal(val, datatype=XSD.double)
    return Literal(str(val))


def _to_literal_or_uri(val: Any) -> URIRef | Literal:
    """Convert a value to URIRef if it looks like a URI, else Literal."""
    if isinstance(val, str) and (
        val.startswith("http://") or val.startswith("https://")
    ):
        return URIRef(val)
    return _to_literal(val)


def _literal_to_python(lit: Literal | URIRef | BNode) -> Any:
    """Convert an rdflib term back to a Python value."""
    if isinstance(lit, URIRef):
        return str(lit)
    if isinstance(lit, BNode):
        return f"_:{lit}"
    if isinstance(lit, Literal):
        dt = lit.datatype
        if dt == XSD.double or dt == XSD.decimal or dt == XSD.float:
            return float(lit)
        if dt == XSD.integer or dt == XSD.int or dt == XSD.long:
            return int(lit)
        if dt == XSD.boolean:
            return lit.toPython()
        return str(lit)
    return str(lit)


def _escape_ntriples_str(s: str) -> str:
    """Escape a string for N-Triples."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace(
        "\n", "\\n"
    ).replace("\r", "\\r").replace("\t", "\\t")


def _format_ntriples_literal(val: Any) -> str:
    """Format a Python value as an N-Triples literal."""
    if isinstance(val, bool):
        bval = "true" if val else "false"
        return f'"{bval}"^^<{XSD}boolean>'
    if isinstance(val, int):
        return f'"{val}"^^<{XSD}integer>'
    if isinstance(val, float):
        return f'"{val}"^^<{XSD}double>'
    escaped = _escape_ntriples_str(str(val))
    return f'"{escaped}"'


def _format_ntriples_annotation(
    val: Any,
    is_iri: bool = False,
    xsd_type: URIRef | None = None,
) -> str:
    """Format an annotation value for N-Triples-Star."""
    if is_iri or (
        isinstance(val, str)
        and (val.startswith("http://") or val.startswith("https://"))
    ):
        return f"<{val}>"
    if isinstance(val, bool):
        bval = "true" if val else "false"
        return f'"{bval}"^^<{XSD}boolean>'
    if isinstance(val, int):
        dt = xsd_type or XSD.integer
        return f'"{val}"^^<{dt}>'
    if isinstance(val, float):
        dt = xsd_type or XSD.double
        return f'"{val}"^^<{dt}>'
    escaped = _escape_ntriples_str(str(val))
    if xsd_type:
        return f'"{escaped}"^^<{xsd_type}>'
    return f'"{escaped}"'


def _parse_nt_term(term: str) -> str:
    """Parse an N-Triples subject/object term to a string."""
    term = term.strip()
    if term.startswith("<") and term.endswith(">"):
        return term[1:-1]
    if term.startswith("_:"):
        return term
    return term


def _parse_nt_value(val_str: str) -> Any:
    """Parse an N-Triples value string to a Python value."""
    val_str = val_str.strip()

    # IRI
    if val_str.startswith("<") and val_str.endswith(">"):
        return val_str[1:-1]

    # Typed literal: "value"^^<type>
    typed_match = re.match(r'^"(.*?)"\^\^<(.+?)>$', val_str, re.DOTALL)
    if typed_match:
        raw, dtype = typed_match.groups()
        raw = raw.replace("\\n", "\n").replace("\\r", "\r").replace(
            "\\t", "\t"
        ).replace('\\"', '"').replace("\\\\", "\\")
        if "double" in dtype or "decimal" in dtype or "float" in dtype:
            return float(raw)
        if "integer" in dtype or "int" in dtype or "long" in dtype:
            return int(raw)
        if "boolean" in dtype:
            return raw.lower() == "true"
        if "dateTime" in dtype:
            return raw
        return raw

    # Plain literal: "value"
    plain_match = re.match(r'^"(.*?)"$', val_str, re.DOTALL)
    if plain_match:
        raw = plain_match.group(1)
        return raw.replace("\\n", "\n").replace("\\r", "\r").replace(
            "\\t", "\t"
        ).replace('\\"', '"').replace("\\\\", "\\")

    # Bare value (shouldn't happen in valid N-Triples)
    return val_str
