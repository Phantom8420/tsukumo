# TSUKUMO 付喪

**The master retires. The tool remembers.**

In Japanese folklore a tool that serves for a hundred years gains a soul (付喪神, *tsukumogami*).
TSUKUMO gives it one in days: a phone watching a master craftsperson's hands (or strapped to the tool) records how they
actually work, finds the moments no manual explains, asks the master *why*, and turns each answer
into a card that coaches apprentices, in their own language, at the same point in the work, while
they work.

The demo craft is whetstone edge finishing (本刃付け) at a fictional Seki knife maker, but nothing in
the pipeline is specific to knives: any hand-tool skill with angle, rhythm, force or sound works.

## The loop

| Stage | What happens | Where |
|---|---|---|
| **Capture** | Three sources, one frame format. *Camera*: a phone on a stand tracks the working hand in the browser (MediaPipe hand landmarks) for tilt and wrist acceleration, plus the mic for the stone's sound. *Video file*: the same tracker run over a recording. *Strapped phone*: motion sensors at ~50 Hz for the most precise angle. The master just works. | `web/phone.html`, `web/js/vision.js` |
| **Ask only about surprises** | A model of "what the master was just doing" (robust rolling median/MAD per channel) flags the moments they broke from it: pauses, angle or pressure changes, plus slow build-ups that preceded them (e.g. the stone's pitch climbing as it dries). Typically 4-6 questions per session. | `tsukumo/surprise.py` |
| **Structure** | The master answers by voice (browser speech recognition, any language). Claude turns the answer plus sensor evidence into a kotsu card (trigger / action / reason) in Japanese, English, Vietnamese and Indonesian. Offline, a rule-based card is built from the master's words and the sensors. | `tsukumo/kotsu.py` |
| **Coach** | Apprentices are aligned to the master *by strokes done, not clock time*. The phone buzzes on sustained drift from the master's ghost (angle, pressure, tempo), and each card fires at the stroke where the master's habit happened. | `tsukumo/coach.py` |
| **Validate** | A finished session is DTW-aligned to the master's reference and scored per channel; weak spots come with concrete advice. Two passing sessions certify the skill. | `tsukumo/align.py` |
| **Plan** | Skill extinction radar (who holds what, when they retire, measured successors, captured cards) and a royalty ledger that pays masters each time their cards coach someone. | `tsukumo/services.py` |

## Real footage

Not every session in the demo is simulated. **Wheel grinding** is a master session measured from
[*Knife sharpening video*](https://commons.wikimedia.org/wiki/File:Knife_sharpening_video.webm) by
Vijayanrajapuram (CC BY-SA 4.0; rotated upright and re-encoded as `web/media/grinder.webm`, which is
shared under the same licence). The clip went through the camera tracker end to end: 46 s became
18 strokes and 6 questions for the master. The strongest of them line up with what you can see in
the clip, such as the wheel's sound dropping 16 dB when the blade lifts off it. The review page plays
the clip and jumps to each question's moment.

What it showed about camera capture, measured rather than assumed:

- **Rate.** CPU hand tracking in a headless browser kept up at about 6.6 frames/s. That is plenty
  for strokes (well under 1 Hz on a wheel) but coarse for acceleration. A phone with GPU inference
  does better.
- **Test-retest.** The same clip played again as an "apprentice" scored **87 %** against itself,
  not 100 %, because tracking lands on different frames each run. Treat score differences under
  about 13 points from camera sessions as noise; use the strapped phone when it needs to be finer.
- **Angle is the hand's tilt**, from the knuckle line against the horizon, not the blade's angle to
  the stone. It moves with the blade but is a proxy, so only compare camera sessions with camera
  sessions.

To process another clip: create a session, then open `phone.html?s=<id>&video=<url>`. It plays the
clip, streams frames and finishes on its own.

## Run it

Requires Python 3.11+.

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"       # Windows; use .venv/bin/pip elsewhere
.venv/Scripts/python -m tsukumo reset        # load the demo workshop
.venv/Scripts/python -m tsukumo serve        # https://localhost:8443
```

`serve` generates a self-signed certificate for the machine's LAN address, because phones only expose
motion sensors and the microphone to secure pages. On the phone, open the URL from the QR code and
accept the certificate warning once. Use `serve --http --port 8765` for a laptop-only demo.

Set `ANTHROPIC_API_KEY` to have Claude structure and translate cards (model `claude-opus-5-5`,
override with `TSUKUMO_MODEL`). Without it, or with `TSUKUMO_OFFLINE=1`, everything else works and
cards are built offline.

## Deploy on Vercel

Import the repo in Vercel; `vercel.json` routes everything to one Python function (`api/index.py`)
that seeds the demo workshop into `/tmp` on cold start (about 7 s). Add `ANTHROPIC_API_KEY` as an
environment variable if you want Claude-structured cards.

Serverless functions can't hold WebSockets, so on Vercel the phone link is off and *Simulate* asks the
server to run the whole session in one request (`POST /api/sessions/{id}/replay`), then plays the
events back in the browser. Each function instance has its own `/tmp` database: edits and new
sessions are per instance and vanish on the next cold start.

## Demo script (3 min)

1. **Workshop** page: two retiring masters, four skills, two with nobody ready to take over.
2. **Review** Tanaka's session: the surprise lane, the questions it raised, two already answered.
   Open Q5 (the micro-bevel), press *Speak*, have the master answer in Japanese (or *Use example*),
   *Make kotsu card*, and the card appears in four languages.
3. **Record & train**: pick Nguyen, start, scan the QR with the phone on the tool (or *Simulate*
   with `novice`). The ghost runs alongside, the phone buzzes in Vietnamese, the master's card pops up
   at stroke 28. End to get the score and weak spots.
   One-link version: `live.html?person=nguyen&skill=honbazuke&sim=novice&speed=2`.
4. **Succession**: the radar, Siti's certified passport, Tanaka's royalties.

## Layout

```
web/js/vision.js  camera/video -> frames: hand landmarks for tilt and wrist acceleration, audio meter
tsukumo/
  config.py     kinds of work: each skill's frame layout and channels
  signals.py    raw frames -> channels (sharpening: angle, force, tempo, sound; heat: colour, glow, soak time)
  surprise.py   change/pause detection with explanations and build-up evidence
  align.py      banded DTW + per-channel scoring + weak spots
  coach.py      online tracker and live coach (buzzes, cards, ghost ticks)
  kotsu.py      Claude card structuring + offline fallback
  simulate.py   synthetic master/apprentice sessions (demo and tests)
  store.py      SQLite metadata, .npy frames
  services.py   use cases: sessions, review, passports, radar, ledger
  live.py       WebSocket hub for sessions being recorded, and HTTP replay
  app.py        FastAPI routes; serves web/
  seed.py       demo workshop
  footage/      frames measured from the real clip
web/            landing (index), dashboard, live, review, radar, phone client
api/index.py    Vercel entry point
tests/          pipeline and API tests
```

Frames are `[t_ms, angle_deg, ax, ay, az, loud_dbfs, tone_hz]`. Anything that produces them (a phone,
an ESP32 with an IMU and contact mic, a smartwatch) can feed `/ws/device/{session_id}`.

## Tests

```bash
.venv/Scripts/python -m pytest -q
```
