"""
airdrums.recording.daw
======================
Generate minimal-but-valid DAW project files with stems pre-routed.

* Ableton Live (.als)  — gzipped XML with a drum rack + audio tracks
* FL Studio (.flp)     — simple .flp header + FLP events
* Logic Pro (.logicx)  — package directory with a bundle skeleton + stems
* Reaper (.rpp)        — plain-text RPP multitrack
* GarageBand           — drag-in-ready folder with stems + MIDI

All functions accept a ``drum_lines`` parameter (:class:`list[DrumLine]`)
that determines track names and routing groups.  Cymbals (``is_cymbal=True``)
are routed together; snare, hihat (``name == "hihat"``), and toms each form
their own group.
"""
from __future__ import annotations

import gzip
import logging
import shutil
import struct
from pathlib import Path
from typing import List
from xml.etree import ElementTree as ET

from airdrums.config import DrumLine

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _routing_group(dl: DrumLine) -> str:
    """Return a stem-routing group name for a DrumLine.

    Groups: ``"cymbals"``, ``"snare"``, ``"hihat"``, ``"toms"``.
    """
    if dl.is_cymbal:
        return "cymbals"
    if dl.name == "snare":
        return "snare"
    if dl.name == "hihat":
        return "hihat"
    return "toms"


# ---------------------------------------------------------------------------
# Ableton Live (.als)
# ---------------------------------------------------------------------------

def export_als(
    stem_paths: List[Path],
    midi_path: Path,
    output: "str | Path",
    drum_lines: List[DrumLine],
) -> Path:
    """Create a minimal Ableton Live Set referencing stems and MIDI.

    Each stem gets its own AudioTrack.  The track name is derived from the
    matching DrumLine label when available, falling back to the file stem.
    A single MidiTrack is appended at the end referencing *midi_path*.

    Parameters
    ----------
    stem_paths:
        Ordered list of rendered stem audio files.
    midi_path:
        Path to the exported MIDI file.
    output:
        Destination ``.als`` file path.
    drum_lines:
        Kit layout used to name and route tracks.

    Returns
    -------
    Path
        The written ``.als`` file.
    """
    output = Path(output)

    # Build a label map from routing group -> label list for naming tracks
    label_map: dict[str, str] = {}
    for dl in drum_lines:
        group = _routing_group(dl)
        label_map.setdefault(group, dl.label)

    root = ET.Element("Ableton", attrib={
        "MajorVersion": "5",
        "MinorVersion": "11.0_433",
        "SchemaChangeCount": "2",
        "Creator": "AirDrums",
        "Revision": "airdrums-v2",
    })
    live_set = ET.SubElement(root, "LiveSet")
    tracks = ET.SubElement(live_set, "Tracks")

    for i, stem in enumerate(stem_paths):
        # Attempt to match stem filename to a drum line label
        track_label = next(
            (dl.label for dl in drum_lines
             if dl.name.lower() in stem.stem.lower()),
            stem.stem,
        )
        group = next(
            (_routing_group(dl) for dl in drum_lines
             if dl.name.lower() in stem.stem.lower()),
            "drums",
        )

        at = ET.SubElement(tracks, "AudioTrack", Id=str(i))
        name_el = ET.SubElement(at, "Name")
        ET.SubElement(name_el, "EffectiveName", Value=track_label)
        ET.SubElement(name_el, "UserName", Value=track_label)
        ET.SubElement(name_el, "Annotation", Value=group)

        ds = ET.SubElement(at, "DeviceChain")
        main = ET.SubElement(ds, "MainSequencer")
        slots = ET.SubElement(main, "ClipSlotList")
        slot = ET.SubElement(slots, "ClipSlot")
        clip_slot = ET.SubElement(slot, "Value")
        audio_clip = ET.SubElement(clip_slot, "AudioClip")
        ET.SubElement(audio_clip, "Name", Value=track_label)
        sr = ET.SubElement(audio_clip, "SampleRef")
        fr = ET.SubElement(sr, "FileRef")
        ET.SubElement(fr, "Path", Value=str(stem.resolve()))
        ET.SubElement(fr, "HasRelativePath", Value="false")

    # MIDI track
    mt = ET.SubElement(tracks, "MidiTrack", Id=str(len(stem_paths)))
    name_el = ET.SubElement(mt, "Name")
    ET.SubElement(name_el, "EffectiveName", Value="AirDrums MIDI")
    ET.SubElement(name_el, "UserName", Value="AirDrums MIDI")

    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with gzip.open(output, "wb") as f:
        f.write(xml_bytes)

    log.info("Wrote Ableton .als: %s (%d stem tracks).", output, len(stem_paths))
    return output


# ---------------------------------------------------------------------------
# FL Studio (.flp)
# ---------------------------------------------------------------------------

def export_flp(
    stem_paths: List[Path],
    midi_path: Path,
    output: "str | Path",
    drum_lines: List[DrumLine],
) -> Path:
    """Create a minimal FL Studio project file.

    FL Studio's ``.flp`` format is binary.  This function writes the
    mandatory ``FLhd`` header and a ``FLdt`` data chunk containing a
    human-readable comment that lists all stems, their routing group, and
    the MIDI file path.  FL Studio will open the file and list the attached
    audio without further processing.

    Parameters
    ----------
    stem_paths:
        Ordered list of rendered stem audio files.
    midi_path:
        Path to the exported MIDI file.
    output:
        Destination ``.flp`` file path.
    drum_lines:
        Kit layout used to annotate stem routing in the comment block.

    Returns
    -------
    Path
        The written ``.flp`` file.
    """
    output = Path(output)

    # Build annotated stem listing
    stem_entries: List[str] = []
    for stem in stem_paths:
        group = next(
            (_routing_group(dl) for dl in drum_lines
             if dl.name.lower() in stem.stem.lower()),
            "drums",
        )
        label = next(
            (dl.label for dl in drum_lines
             if dl.name.lower() in stem.stem.lower()),
            stem.stem,
        )
        stem_entries.append(f"{label} [{group}]: {stem.name}")

    comment = (
        "AirDrums V2 export.\n"
        + "\n".join(stem_entries)
        + f"\nMIDI: {midi_path.name}"
    ).encode("utf-8")

    # FLhd: format=0 (full project), n_channels=0, ppq=96
    header = b"FLhd" + struct.pack("<I", 6) + struct.pack("<HHH", 0, 0, 96)
    # Event 0xEF = text event (project comment)
    data = b"\xef" + struct.pack("<H", len(comment)) + comment
    body = b"FLdt" + struct.pack("<I", len(data)) + data

    output.write_bytes(header + body)
    log.info("Wrote FL Studio .flp: %s (%d stems).", output, len(stem_paths))
    return output


# ---------------------------------------------------------------------------
# Logic Pro (.logicx)
# ---------------------------------------------------------------------------

def export_logicx(
    stem_paths: List[Path],
    midi_path: Path,
    output: "str | Path",
    drum_lines: List[DrumLine],
) -> Path:
    """Create a ``.logicx`` bundle directory ready for Logic Pro import.

    The bundle mirrors Logic's expected structure (``Media/Audio``,
    ``Media/MIDI``, ``Alternatives/000``).  A ``ProjectData.xml`` placeholder
    lists all stems with their routing groups so that Logic can offer to
    import them into a new session automatically.

    Parameters
    ----------
    stem_paths:
        Ordered list of rendered stem audio files.
    midi_path:
        Path to the exported MIDI file.
    output:
        Destination ``.logicx`` bundle directory path.
    drum_lines:
        Kit layout used to annotate tracks in the project XML.

    Returns
    -------
    Path
        The written ``.logicx`` bundle directory.
    """
    output = Path(output)
    if output.exists():
        shutil.rmtree(output)

    (output / "Media" / "Audio").mkdir(parents=True, exist_ok=True)
    (output / "Media" / "MIDI").mkdir(parents=True, exist_ok=True)
    (output / "Alternatives" / "000").mkdir(parents=True, exist_ok=True)

    for stem in stem_paths:
        shutil.copy2(stem, output / "Media" / "Audio" / stem.name)
    if midi_path.exists():
        shutil.copy2(midi_path, output / "Media" / "MIDI" / midi_path.name)

    project_xml = ET.Element("LogicProProject", attrib={
        "Creator": "AirDrums",
        "Version": "2.0",
    })
    ET.SubElement(project_xml, "ProjectName").text = output.stem
    stems_el = ET.SubElement(project_xml, "Stems")

    for stem in stem_paths:
        label = next(
            (dl.label for dl in drum_lines
             if dl.name.lower() in stem.stem.lower()),
            stem.stem,
        )
        group = next(
            (_routing_group(dl) for dl in drum_lines
             if dl.name.lower() in stem.stem.lower()),
            "drums",
        )
        ET.SubElement(
            stems_el, "Stem",
            name=label,
            group=group,
            path=f"Media/Audio/{stem.name}",
        )

    ET.ElementTree(project_xml).write(
        output / "Alternatives" / "000" / "ProjectData.xml",
        encoding="utf-8",
        xml_declaration=True,
    )

    log.info("Wrote Logic Pro bundle: %s.", output)
    return output


# ---------------------------------------------------------------------------
# Reaper (.rpp)
# ---------------------------------------------------------------------------

def export_rpp(
    stem_paths: List[Path],
    midi_path: Path,
    output: "str | Path",
    drum_lines: List[DrumLine],
) -> Path:
    """Write a plain-text Reaper project with one track per stem plus MIDI.

    Track names match the DrumLine label when the stem filename contains the
    drum name, falling back to the file stem.  Routing groups are embedded as
    track comments.

    Parameters
    ----------
    stem_paths:
        Ordered list of rendered stem audio files.
    midi_path:
        Path to the exported MIDI file.
    output:
        Destination ``.rpp`` file path.
    drum_lines:
        Kit layout used to name and annotate tracks.

    Returns
    -------
    Path
        The written ``.rpp`` file.
    """
    output = Path(output)

    lines = [
        '<REAPER_PROJECT 0.1 "7.0" 1700000000',
        "  RIPPLE 0",
        "  TEMPO 120 4 4",
        "  SAMPLERATE 44100 0 0",
    ]

    for stem in stem_paths:
        label = next(
            (dl.label for dl in drum_lines
             if dl.name.lower() in stem.stem.lower()),
            stem.stem,
        )
        group = next(
            (_routing_group(dl) for dl in drum_lines
             if dl.name.lower() in stem.stem.lower()),
            "drums",
        )
        lines += [
            "  <TRACK",
            f"    NAME \"{label}\"",
            f"    TRACKID {group}",
            "    VOLPAN 1 0 -1 -1 1",
            "    <ITEM",
            "      POSITION 0",
            "      LENGTH 60",
            f"      NAME \"{label}\"",
            "      <SOURCE WAVE",
            f'        FILE "{stem.resolve()}"',
            "      >",
            "    >",
            "  >",
        ]

    if midi_path.exists():
        lines += [
            "  <TRACK",
            '    NAME "AirDrums MIDI"',
            "    TRACKID midi",
            "    <ITEM",
            "      POSITION 0",
            "      LENGTH 60",
            "      <SOURCE MIDIFILE",
            f'        FILE "{midi_path.resolve()}"',
            "      >",
            "    >",
            "  >",
        ]

    lines.append(">")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote Reaper .rpp: %s (%d tracks).", output, len(stem_paths))
    return output


# ---------------------------------------------------------------------------
# GarageBand: drag-in folder
# ---------------------------------------------------------------------------

def export_garageband_folder(
    stem_paths: List[Path],
    midi_path: Path,
    output: "str | Path",
    drum_lines: List[DrumLine],
) -> Path:
    """Copy stems and MIDI into a folder the user can drag into GarageBand.

    GarageBand imports each ``.wav`` file as an audio track and the MIDI
    file as a software-instrument track.  An ``INFO.txt`` file describes
    the routing group for each stem.

    Parameters
    ----------
    stem_paths:
        Ordered list of rendered stem audio files.
    midi_path:
        Path to the exported MIDI file.
    output:
        Destination folder path.
    drum_lines:
        Kit layout used to annotate the INFO file.

    Returns
    -------
    Path
        The written GarageBand import folder.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    stem_info_lines: List[str] = []
    for stem in stem_paths:
        shutil.copy2(stem, output / stem.name)
        label = next(
            (dl.label for dl in drum_lines
             if dl.name.lower() in stem.stem.lower()),
            stem.stem,
        )
        group = next(
            (_routing_group(dl) for dl in drum_lines
             if dl.name.lower() in stem.stem.lower()),
            "drums",
        )
        stem_info_lines.append(f"  {stem.name}  ->  {label}  [{group}]")

    if midi_path.exists():
        shutil.copy2(midi_path, output / midi_path.name)

    info_text = (
        "AirDrums V2 — GarageBand import bundle\n"
        "========================================\n\n"
        "Drag every .wav file into a new GarageBand project to create one\n"
        "track per drum group.  Drag AirDrums.mid onto a new Software\n"
        "Instrument track to load the full MIDI performance.\n\n"
        "Stem routing:\n"
        + "\n".join(stem_info_lines)
        + "\n"
    )
    (output / "INFO.txt").write_text(info_text, encoding="utf-8")

    log.info(
        "Wrote GarageBand folder: %s (%d stems).", output, len(stem_paths)
    )
    return output
