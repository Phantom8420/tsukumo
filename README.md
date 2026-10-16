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
51 strokes and 5 questions for the master. The review page plays the clip and jumps to each
question's moment.

**Checked against labels, not the simulator.** Watching the clip, the blade lifts off the wheel
three times (about 14.6, 24.5 and 42.9 s) and the camera is moved between 33 and 36 s. The labels
are in `tests/test_pipeline.py`, which fails if they stop matching.

| | Lifts asked about | False alarms on the camera move |
|---|---|---|
| Clip played in real time (first version) | 2 of 3 | 1 |
| Clip stepped frame by frame (now) | **3 of 3** | **0** |

The other two questions (6 s, 18 s) can't be confirmed or ruled out by eye. Three labelled events
is a small sample.

What it showed about camera capture, measured rather than assumed:

- **Recorded clips are stepped, not played.** The tracker seeks to every 1/15 s and reads each
  frame afresh (MediaPipe image mode, GPU with CPU fallback); the sound is read from the file, not
  the speakers. Played in real time, the tracker dropped frames and its frame-to-frame state drifted:
  the same clip gave 18 strokes and scored 72 to 87 % against itself. Stepped, two master
  recordings gave 51 and 52 strokes and the same questions to within a second, and the clip scored
  **96.5, 96.8 and 98.3 %** against itself. Scores from clips carry a ±4 margin.
- **Live camera is still real time**, so it keeps the old margin: the same clip played live scored
  87 % against itself. Live camera scores show ±13, and a filmed pass needs the mark plus that
  margin. Use the strapped phone when it needs to be finer.
- **Stroke counts depend on frame rate.** At 15 frames/s the tracker sees 51 strokes where real-time
  playback (about 6.6 frames/s on a headless CPU) saw 18. The wrist turns are too small to count by
  eye in the clip, so neither number is verified.
- **Angle is the hand's tilt**, from the knuckle line against the horizon, not the blade's angle to
  the stone. It moves with the blade but is a proxy, so only compare camera sessions with camera
  sessions.

To process another clip: create a session, then open `phone.html?s=<id>&video=<url>`. It plays the
clip frame by frame, streams frames and finishes on its own.

## Second craft: quench hardening

To show the pipeline isn't knife-sharpening code, **Quench hardening** (焼入れ) runs on different
channels: the steel's colour and glow (from the camera), soak time at heat, and hand movement.
Yamada's master session has two answered questions: a turn of the blade halfway through the soak and
a pull from the fire to check its colour. Both were found by the same surprise model, without being
told what to look for. The simulator gives these scores against Yamada over three seeds: master 100,
skilled 92-93, novice 67-68. The novice soaks too short and runs hot, so it falls below the 75 pass mark.
Only time at heat is scored. Before the fire and after the quench everyone's steel is dark, and
counting those frames let a novice reach 80.

One-link version: `live.html?person=nguyen&skill=yakiire&sim=novice&speed=4`.

**From the camera.** On a quench session the phone page reads colour instead of the hand: tap the
steel on the preview, and it takes hue and glow from the brightest coloured pixels in that box
(a white wall or a grey bench reads as 0, not as glow). Exposure and white balance are locked
after 1.5 s where the browser allows it, because auto exposure would "correct" the very colour
being measured; the page says whether the lock took. Checked on clips rendered from the simulator:
a filmed master scored **98** against itself and the filmed novice **70**; the master's glow rise
and its check (glow down 29 %, colour down 7°) were found as questions. The turn was not, because
those clips have no hand in them, and movement comes from a visible hand.

Limits, measured: every camera and codec sees colour its own way (the same red decoded 10° apart
after a VP9 round-trip), so absolute hue means nothing across devices. Hence the rule below.

**Record the way the master did.** An apprentice is only compared with a master recorded the same
way: filmed with filmed, strapped phone with strapped phone. Otherwise one side measures the hand's
tilt and the other the blade's angle, or two cameras' colour. The phone page stops with a message
when they differ. The seeded masters wore the phone, so practise against them with the phone or the
simulator; to train by camera, film a master on the same rig and set it as the reference.

## Does it find the right moments?

`python -m tsukumo eval` plants the master's habits at known times in 20 simulated sessions, then
checks whether a question starts within 2.5 s of each one. It repeats this with the same habits
done by less steady hands:

| Hands | Sharpening: found | Quench: found | False alarms |
|---|---|---|---|
| Master | water, burr, micro-bevel 100 % · flip 0 % | turn, check, quench 100 % | 0 % |
| Skilled-level unsteadiness | same | same | 0 % |
| Novice-level unsteadiness (3° drift) | water, burr 100 % · **micro-bevel 30 %** · flip 0 % | 100 % | 12 % (sharpening) |

What that means:

- **Pauses and big changes are reliable.** Once the hand wanders as much as the habit changes
  (3° drift against a 2.5° micro-bevel), the small change is lost in it. Capture masters, not
  trainees, which is the point of the product anyway.
- **The flip is never asked about.** Angle is stored without its sign so that an apprentice on
  either face matches the master, and so a change of face looks like nothing happened. Its card
  would have to be added by hand.
- **The checks and the quench share one question.** They are 1.2 s apart, and the model keeps
  questions at least 4 s apart.
- These numbers come from the simulator, which plants clean habits. The real grinder clip is
  checked against hand-labelled lifts instead (see *Real footage*).

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

Deleting sessions or cards and changing a skill's master can't be undone. With `TSUKUMO_KEY` set,
those need the workshop key, which the browser asks for once. Without it, as on the public demo,
visitors can undo their own work but can't touch the seeded workshop or the real footage.

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
  evaluate.py   planted-habit recall/precision for surprise detection (`python -m tsukumo eval`)
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
