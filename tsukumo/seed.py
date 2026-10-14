"""Demo workshop: a Seki knife maker with two retiring masters and three apprentices.

Everything here is fictional. The master's sharpening session and the
apprentices' sessions are simulated, then pushed through the real pipeline
(surprise detection, scoring), so what the demo shows is what the system
actually computes.
"""

from __future__ import annotations

import time

from .services import Tsukumo
from .simulate import generate

PEOPLE = [
    dict(id="tanaka", name="Tanaka Ken'ichi", name_native="田中 健一", role="master", lang="ja",
         retire_year=2027, origin="Seki, Gifu", bio="Edge finisher, 41 years at the bench."),
    dict(id="yamada", name="Yamada Akio", name_native="山田 昭夫", role="master", lang="ja",
         retire_year=2026, origin="Seki, Gifu", bio="Heat treatment, 46 years. Reads steel by colour."),
    dict(id="nguyen", name="Nguyen Van An", name_native="Nguyễn Văn An", role="apprentice", lang="vi",
         origin="Nam Dinh, Vietnam", bio="Specified Skilled Worker, second year."),
    dict(id="siti", name="Siti Rahmawati", name_native="Siti Rahmawati", role="apprentice", lang="id",
         origin="Bandung, Indonesia", bio="Specified Skilled Worker, third year."),
    dict(id="sato", name="Sato Yui", name_native="佐藤 結衣", role="apprentice", lang="ja",
         origin="Gifu", bio="Joined after technical high school."),
]

SKILLS = [
    dict(id="honbazuke", name="Final edge sharpening", name_ja="本刃付け",
         description="Whetstone finishing of the edge: angle, rhythm, burr, micro-bevel."),
    dict(id="yakiire", name="Quench hardening", name_ja="焼入れ",
         description="Heating to temperature by colour and quenching for hardness."),
    dict(id="hizukuri", name="Forge shaping", name_ja="火造り",
         description="Hammering the blank to shape at the forge."),
    dict(id="etsuke", name="Handle fitting", name_ja="柄付け",
         description="Fitting and setting the handle."),
]

# Declared, not measured: what the workshop already knows about who can do what.
PROFICIENCY = [
    ("yamada", "yakiire", 1.0), ("yamada", "hizukuri", 0.9),
    ("tanaka", "hizukuri", 0.6),
    ("sato", "etsuke", 0.85), ("siti", "etsuke", 0.8), ("nguyen", "hizukuri", 0.5),
]

# Cards the master already recorded for two of the habits the sensors caught.
# The micro-bevel surprise is left open so it can be answered live in a demo.
CARDS = {
    "water": {
        "transcript": "音が高く、カサカサしてきたら砥石が乾いてる証拠。手を止めて水をかける。"
                      "乾いた砥石だと研ぎ汁がなくなって、刃が引っかかって熱を持つ。研いでるのは泥なんだ。",
        "content": {
            "ja": {"title": "砥石の「乾いた音」を聞く",
                   "trigger": "研ぐ音が高く、カサカサしてきたとき",
                   "action": "手を止めて、砥石に水をかける",
                   "reason": "乾いた砥石は研ぎ汁(泥)がなくなり、刃が引っかかって熱を持つ。実際に研いでいるのは泥。"},
            "en": {"title": "Listen for the stone going dry",
                   "trigger": "When the sharpening sound turns higher and scratchy",
                   "action": "Stop and splash water on the stone",
                   "reason": "A dry stone loses its slurry, so the blade catches and heats up. The slurry is what actually sharpens."},
            "vi": {"title": "Lắng nghe tiếng đá mài bị khô",
                   "trigger": "Khi tiếng mài trở nên cao và rít hơn",
                   "action": "Dừng tay và tưới nước lên đá mài",
                   "reason": "Đá khô sẽ mất lớp bùn mài, lưỡi dao bị vướng và nóng lên. Chính lớp bùn mới là thứ mài sắc."},
            "id": {"title": "Dengarkan batu asah yang mulai kering",
                   "trigger": "Saat suara asahan menjadi lebih tinggi dan kasar",
                   "action": "Berhenti dan siram batu asah dengan air",
                   "reason": "Batu yang kering kehilangan lumpur asahnya, pisau tersangkut dan menjadi panas. Lumpur itulah yang sebenarnya mengasah."},
        },
    },
    "burr": {
        "transcript": "裏返す前に、親指で刃先のカエリを確かめる。刃元から切っ先まで全部出てたら裏に返す。"
                      "出てない所はまだ刃が付いてない。早く返すとそこだけ切れない。",
        "content": {
            "ja": {"title": "返す前にカエリを指で確かめる",
                   "trigger": "片面をしばらく研いだら",
                   "action": "手を止め、親指で刃先から外へ撫でてカエリを確かめる。刃元から切っ先まで出ていたら裏返す。",
                   "reason": "カエリが出ていない所はまだ刃が付いていない。早く返すと、そこだけ切れない刃になる。"},
            "en": {"title": "Feel for the burr before you flip",
                   "trigger": "After a stretch of strokes on one face",
                   "action": "Stop. Brush your thumb off the edge to feel for the burr. Flip only when it runs heel to tip.",
                   "reason": "Where there's no burr, the edge hasn't been reached yet. Flip early and that section stays dull."},
            "vi": {"title": "Sờ ba via trước khi lật mặt",
                   "trigger": "Sau một loạt nhát mài trên một mặt",
                   "action": "Dừng lại. Vuốt ngón cái ra khỏi lưỡi để cảm nhận ba via. Chỉ lật khi ba via chạy suốt từ gót đến mũi dao.",
                   "reason": "Chỗ nào chưa có ba via là chỗ đó chưa mài tới cạnh. Lật sớm thì đoạn đó vẫn cùn."},
            "id": {"title": "Raba burr sebelum membalik",
                   "trigger": "Setelah beberapa lama mengasah satu sisi",
                   "action": "Berhenti. Usapkan ibu jari menjauhi mata pisau untuk merasakan burr. Balik hanya jika burr ada dari pangkal sampai ujung.",
                   "reason": "Bagian tanpa burr berarti mata pisau belum tercapai. Membalik terlalu cepat membuat bagian itu tetap tumpul."},
        },
    },
}

# (apprentice, profile, seed, days ago)
APPRENTICE_RUNS = [
    ("nguyen", "novice", 11, 9), ("nguyen", "novice", 12, 5), ("nguyen", "skilled", 13, 1),
    ("siti", "skilled", 21, 8), ("siti", "skilled", 22, 3),
    ("sato", "novice", 31, 6), ("sato", "skilled", 32, 2),
]


def _match(surprises: list[dict], events: list[dict]) -> dict[str, str]:
    """surprise id -> habit, by nearest time, so seeded cards attach to the right moment."""
    out = {}
    for s in surprises:
        near = min(events, key=lambda e: abs(e["t"] - s["t_start"]), default=None)
        if near and abs(near["t"] - s["t_start"]) < 2.5:
            out[s["id"]] = near["habit"]
    return out


def seed(service: Tsukumo) -> dict:
    store = service.store
    if store.skills():
        return {"seeded": False}
    for p in PEOPLE:
        store.upsert_person(**p)
    for s in SKILLS:
        store.upsert_skill(**s)
    for person, skill, level in PROFICIENCY:
        store.set_proficiency(person, skill, level)

    now = time.time()
    day = 86400.0
    master = service.create_session("honbazuke", "tanaka", source="seed", created_at=now - 12 * day)
    frames, events = generate("master", seed=0)
    service.finalize(master["id"], frames)

    surprises = store.surprises(master["id"])
    habits = _match(surprises, events)
    answered: set[str] = set()
    card_ids = []
    for s in surprises:
        habit = habits.get(s["id"])
        if habit in CARDS and habit not in answered:
            spec = CARDS[habit]
            card = store.add_card(skill_id="honbazuke", master_id="tanaka", stroke_index=s["stroke_index"],
                                  content=spec["content"], source="seed", session_id=master["id"],
                                  surprise_id=s["id"], transcript=spec["transcript"], transcript_lang="ja",
                                  evidence={"kind": s["kind"], "summary": s["summary"], "t_start": s["t_start"]})
            store.set_surprise_status(master["id"], s["id"], "answered", card["id"])
            answered.add(habit)
            card_ids.append(card["id"])
        elif habit in CARDS:
            store.set_surprise_status(master["id"], s["id"], "dismissed", f"Same habit as an earlier card ({habit})")

    for person, profile, sd, days in APPRENTICE_RUNS:
        ses = service.create_session("honbazuke", person, source="seed", created_at=now - days * day)
        frames, _ = generate(profile, seed=sd)
        service.finalize(ses["id"], frames)
        for cid in card_ids:
            store.log_card_view(cid, person, ses["id"], at=now - days * day)

    return {"seeded": True, "master_session": master["id"]}
