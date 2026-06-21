import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pianoplayer import core
from pianoplayer.hand import Hand
from pianoplayer.models import INote
from pianoplayer.musicxml_io import noteseq_from_part, parse_musicxml
from pianoplayer.vkeyboard import note_name


def _note(idx: int, measure: int = 1, **kwargs) -> INote:
    values = dict(
        name="C",
        pitch=60 + idx,
        octave=4,
        x=float(idx),
        time=float(idx),
        duration=1.0,
        measure=measure,
    )
    values.update(kwargs)
    return INote(**values)


def test_start_measure_one_processes_exactly_one_measure() -> None:
    notes = [_note(0, 1), _note(1, 2)]
    hand = Hand(notes, side="right")
    hand.verbose = False
    hand.optimize_seq = lambda *_: ([1, 2, 3, 4, 5, 1, 2, 3, 4], 0.0)  # type: ignore[method-assign]

    hand.generate(start_measure=1, nmeasures=1)

    assert notes[0].fingering
    assert notes[1].fingering == 0


def test_multivoice_measure_uses_furthest_voice_endpoint(tmp_path: Path) -> None:
    xml = """<score-partwise><part id="P1">
      <measure number="1"><attributes><divisions>1</divisions></attributes>
        <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
        <backup><duration>4</duration></backup>
        <note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration></note>
      </measure>
      <measure number="2">
        <note><pitch><step>G</step><octave>4</octave></pitch><duration>1</duration></note>
      </measure>
    </part></score-partwise>"""
    path = tmp_path / "voices.xml"
    path.write_text(xml, encoding="utf-8")

    score = parse_musicxml(str(path))

    measure_two = [event for event in score.parts[0].events if event.measure == 2]
    assert measure_two[0].offset == 4.0


def test_mixed_tie_chord_keeps_fresh_attacks(tmp_path: Path) -> None:
    xml = """<score-partwise><part id="P1"><measure number="1">
      <attributes><divisions>1</divisions></attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration>
        <tie type="stop"/></note>
      <note><chord/><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration></note>
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>1</duration></note>
    </measure></part></score-partwise>"""
    path = tmp_path / "ties.xml"
    path.write_text(xml, encoding="utf-8")

    score = parse_musicxml(str(path))
    seq = noteseq_from_part(score.parts[0])

    assert [note.pitch for note in seq] == [64, 67]


def test_explicit_chord_merges_with_simultaneous_note_from_another_voice(
    tmp_path: Path,
) -> None:
    xml = """<score-partwise><part id="P1"><measure number="1">
      <attributes><divisions>1</divisions></attributes>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration>
        <voice>1</voice><staff>1</staff></note>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>1</duration>
        <voice>1</voice><staff>1</staff></note>
      <backup><duration>2</duration></backup>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration>
        <voice>2</voice><staff>1</staff></note>
      <note><chord/><pitch><step>G</step><octave>4</octave></pitch><duration>1</duration>
        <voice>2</voice><staff>1</staff></note>
    </measure></part></score-partwise>"""
    path = tmp_path / "mixed-chord-voices.xml"
    path.write_text(xml, encoding="utf-8")

    score = parse_musicxml(str(path))
    seq = noteseq_from_part(score.parts[0])
    simultaneous = [note for note in seq if note.pitch in {60, 67, 72}]

    assert len({note.chordID for note in simultaneous}) == 1
    assert all(note.isChord and note.NinChord == 3 for note in simultaneous)

    hand = Hand(seq, side="right")
    hand.verbose = False
    hand.autodepth = False
    hand.depth = 5
    hand.optimize_seq = lambda *_: ([5] * 9, 0.0)  # type: ignore[method-assign]
    hand.generate(start_measure=1, nmeasures=1)

    assert len({note.fingering for note in simultaneous}) == 3


def test_single_note_and_default_namespace_are_supported(tmp_path: Path) -> None:
    xml = """<score-partwise xmlns="http://www.musicxml.org/ns/musicxml" version="4.0">
      <part id="P1"><measure number="1">
        <attributes><divisions>1</divisions></attributes>
        <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
      </measure></part>
    </score-partwise>"""
    path = tmp_path / "single-namespaced.xml"
    path.write_text(xml, encoding="utf-8")

    score = parse_musicxml(str(path))

    assert len(noteseq_from_part(score.parts[0])) == 1


def test_chord_correction_updates_finger_position_state() -> None:
    notes = [
        _note(0, isChord=True, chordID=7, chordnr=0, NinChord=2),
        _note(4, isChord=True, chordID=7, chordnr=1, NinChord=2),
    ]
    hand = Hand(notes, side="right")
    hand.verbose = False
    hand.autodepth = False
    hand.depth = 5
    hand.optimize_seq = lambda *_: ([1] * 9, 0.0)  # type: ignore[method-assign]

    hand.generate(start_measure=1, nmeasures=1)

    assert notes[1].fingering == 2
    assert hand.fingerseq[1][2] == notes[1].x


def test_conflicting_hand_only_options_are_rejected() -> None:
    args = SimpleNamespace(
        hand_size="M",
        left_only=True,
        right_only=True,
        quiet=True,
        depth=0,
        below_beam=False,
        start_measure=1,
        n_measures=1,
    )
    with pytest.raises(ValueError, match="cannot be used together"):
        core.generate_hands(args, [], [])


def test_musescore_uppercase_extension_gets_xml_conversion_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "SCORE.MSCZ"
    source.write_bytes(b"placeholder")
    seen = {}
    monkeypatch.setattr(core, "_run_external", lambda cmd, _context: seen.setdefault("cmd", cmd))
    monkeypatch.setattr(core, "parse_musicxml", lambda filename: SimpleNamespace(parts=[]))
    args = SimpleNamespace(
        filename=str(source),
        left_only=False,
        right_only=True,
        rpart=0,
        lpart=1,
        rstaff=0,
        lstaff=0,
        auto_routing=True,
        chord_note_stagger_s=0.05,
    )

    xmlfn, *_ = core.load_note_sequences(args)

    assert xmlfn.endswith("SCORE.xml")
    assert seen["cmd"][-1].endswith("SCORE.xml")


def test_single_instrument_midi_auto_routes_without_index_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "one.mid"
    source.write_bytes(b"placeholder")
    fake_midi = SimpleNamespace(instruments=[SimpleNamespace(notes=[])])
    monkeypatch.setitem(
        sys.modules,
        "pretty_midi",
        SimpleNamespace(PrettyMIDI=lambda _filename: fake_midi),
    )
    args = SimpleNamespace(
        filename=str(source),
        left_only=False,
        right_only=False,
        rpart=0,
        lpart=1,
        auto_routing=True,
        chord_note_stagger_s=0.05,
    )

    _, _, right, left = core.load_note_sequences(args)

    assert right == []
    assert left is None
    assert args.right_only is True


def test_single_staff_musicxml_auto_routes_to_one_hand(tmp_path: Path) -> None:
    xml = """<score-partwise><part id="P1"><measure number="1">
      <attributes><divisions>1</divisions></attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration>
        <staff>1</staff></note>
    </measure></part></score-partwise>"""
    path = tmp_path / "single_staff.xml"
    path.write_text(xml, encoding="utf-8")
    args = SimpleNamespace(
        filename=str(path),
        left_only=False,
        right_only=False,
        rpart=0,
        lpart=1,
        rstaff=0,
        lstaff=0,
        auto_routing=True,
        chord_note_stagger_s=0.05,
    )

    _, _, right, left = core.load_note_sequences(args)

    assert len(right) == 1
    assert left is None
    assert args.right_only is True


def test_pig_default_output_and_cost_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("0 0.0 1.0 C4 0 0 0 _\n", encoding="utf-8")
    costs = tmp_path / "costs.csv"
    monkeypatch.chdir(tmp_path)

    core.run_annotate(
        filename=str(source),
        n_measures=1,
        start_measure=1,
        right_only=True,
        quiet=True,
        cost_path=str(costs),
    )

    output = tmp_path / "output.txt"
    assert output.exists() and output.read_text(encoding="utf-8").strip()
    with costs.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0][:3] == ["hand", "index", "time"]
    assert rows[1][0] == "RH"


def test_default_output_and_enharmonic_octaves() -> None:
    assert core.default_output_filename("song.mid") == "output.txt"
    assert core.default_output_filename("song.xml") == "output.xml"
    assert note_name(INote(name="C-", octave=4, pitch=59)) == "B3"
    assert note_name(INote(name="B#", octave=4, pitch=72)) == "C5"
