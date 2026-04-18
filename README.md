# AirDrums

Turn any webcam into a full drum kit. MediaPipe Pose for skeleton tracking,
Depth Anything V2 for per-pixel depth, a virtual drumstick model where only
the tip triggers hits, foot-pedal detection from heel + toe landmarks, live
MIDI output to any DAW, and full session export (MIDI, WAV, stems, BVH mocap,
analytics PDF, DAW projects).

Hackathon build. Single-file install, three hardware profiles, runs on
CPU-only laptops, Apple Silicon, and NVIDIA GPUs.

## Install

```bash
git clone https://github.com/yourname/airdrums
cd airdrums
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional: MP3 export needs ffmpeg on your PATH
#   macOS:   brew install ffmpeg
#   Debian:  sudo apt install ffmpeg
#   Windows: choco install ffmpeg
```

## Run

```bash
python -m airdrums.main                 # auto-detect profile
python -m airdrums.main --profile high  # force NVIDIA path
python -m airdrums.main --no-depth      # pure pose, no DA-V2
python -m airdrums.main --calibrate     # force recalibration
```

### Keyboard controls

| Key   | Action                                   |
|-------|------------------------------------------|
| `q`   | Quit                                     |
| `r`   | Toggle recording                         |
| `o`   | Toggle overdub                           |
| `l`   | Cycle loop length (1 / 2 / 4 / 8 bars)   |
| `t`   | New take                                 |
| `u`   | Undo last take                           |
| `d`   | Toggle depth-map overlay                 |
| `c`   | Re-run calibration                       |
| `[` / `]` | Shrink / grow drumstick length       |
| `s`   | Open PyQt5 settings panel                |
| `e`   | Export session (all formats)             |
| `h`   | Cycle HUD theme (dark / light / neon)    |

## Sample packs

Drop WAV files at `packs/<pack_name>/`:

```
packs/
  default/
    kick_mf.wav
    kick_ff.wav
    snare_pp.wav   snare_mf.wav   snare_ff.wav
    hihat_closed_mf.wav
    hihat_open_mf.wav
    hihat_half_mf.wav
    hihat_chick.wav
    tom1_mf.wav   tom2_mf.wav   floor_tom_mf.wav
    crash_cymbal_mf.wav
    ride_cymbal_mf.wav
```

Velocity layers are auto-detected from the trailing suffix
(`_pp`, `_mp`, `_mf`, `_f`, `_ff`). Multiple files per drum/layer get
round-robin cycling so rolls don't sound machine-gunned.

## DAW connection

AirDrums opens a virtual MIDI port named **AirDrums**. Any DAW that scans
MIDI inputs at startup will see it.

- **Ableton Live**: Preferences → Link/Tempo/MIDI → MIDI → enable Track + Sync
  on "AirDrums". Arm a MIDI track; set input to AirDrums channel 10.
- **Logic Pro / GarageBand**: nothing to configure. Arm a software instrument
  track with any drum kit and the input is live.
- **FL Studio**: Options → MIDI Settings → enable "AirDrums" as input, map
  channel 10 to the FPC/kit you want.
- **Reaper**: Preferences → MIDI → enable "AirDrums" for input, then arm a
  track and set Input → MIDI → AirDrums → All Channels.

### Windows note

Windows does not expose true virtual MIDI ports. Install
[**loopMIDI**](https://www.tobias-erichsen.de/software/loopmidi.html), create
a loopback named "AirDrums", and AirDrums will hook into it automatically.

## Exports

Pressing `e` writes a timestamped folder under `~/.airdrums/exports/`
containing:

- `AirDrums.mid` - type-1 MIDI with tempo map + optional quantization
- `mix.wav` + `mix.mp3` - full mixdown, normalized to -1 dBFS
- `stems/stem_kick.wav` etc. - per-drum stems
- `motion.bvh` - motion-capture of your skeleton (Blender / Maya / MotionBuilder)
- `report.pdf` - analytics report with heatmaps and velocity histograms
- `analytics/*.png` - individual heatmaps and histograms
- `project.als / .flp / .logicx / .rpp` - DAW projects with stems pre-routed
- `GarageBand/` - drag-in folder with stems + MIDI
- `session.json` - reimportable full session data

## Architecture

```
airdrums/
  main.py              Entry point + CLI + main loop
  config.py            Constants, zone coords, MIDI map, profiles, paths
  calibration.py       4-stage calibration
  tracking/
    skeleton.py        MediaPipe Pose + depth sampling + velocity history
    depth.py           Threaded Depth Anything V2
    detectors.py       Velocity-spike + foot-pedal state machine
    drumstick.py       Virtual drumstick model (tip = strike point)
  audio/
    engine.py          pygame velocity-layer sample playback
    export.py          WAV / MP3 / stems offline render
  midi/
    output.py          Virtual MIDI port + 24 PPQN clock
    export.py          Type-1 MIDI file writer
  recording/
    session.py         Events, keyframes, takes, loop/overdub
    bvh.py             Motion capture export
    daw.py             .als / .flp / .logicx / .rpp / GarageBand
  ui/
    overlay.py         Live HUD (zones, skeleton, sticks, FX, BPM, status)
    settings.py        PyQt5 settings panel
  analytics/
    stats.py           Live BPM + post-session heatmaps / timing / PDF
```

## License

MIT.
