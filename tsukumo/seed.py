"""Demo workshop: a Seki knife maker with two retiring masters and three apprentices.

The workshop is fictional. Its sharpening sessions are simulated, then
pushed through the real pipeline (surprise detection, scoring), so what the
demo shows is what the system actually computes. One master is not simulated:
the wheel-grinding session is real Wikimedia Commons footage that the camera
tracker (web/js/vision.js) stepped through frame by frame, stored in footage/grinder.npy.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .services import Tsukumo
from .simulate import generate, generate_heat

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
    dict(id="grinder", name="Street knife-grinder", role="master", lang="en",
         origin="Wikimedia Commons footage", bio="Real, unscripted footage of a pedal-wheel grinder at work."),
]

FOOTAGE = Path(__file__).parent / "footage" / "grinder.npy"

SKILLS = [
    dict(id="honbazuke", name="Final edge sharpening", name_ja="本刃付け",
         description="Whetstone finishing of the edge: angle, rhythm, burr, micro-bevel."),
    dict(id="yakiire", name="Quench hardening", name_ja="焼入れ",
         description="Heating to temperature by colour and quenching for hardness.", kind="heat"),
    dict(id="hizukuri", name="Forge shaping", name_ja="火造り",
         description="Hammering the blank to shape at the forge."),
    dict(id="etsuke", name="Handle fitting", name_ja="柄付け",
         description="Fitting and setting the handle."),
    dict(id="kaiten", name="Wheel grinding", name_ja="回転砥石研ぎ",
         description="Edge grinding on a pedal-driven wheel, from real camera footage."),
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

# Yamada's answers for the two forge habits the sensors caught.
HEAT_CARDS = {
    "turn": {
        "transcript": "色が揃ったら、半分のところで一度刃を返す。火の中で峰の方が遅れるから、返さないと峰だけ焼きが甘くなる。",
        "content": {
            "ja": {"title": "沸かしの途中で一度返す",
                   "trigger": "色が揃ってきて、保持時間の半分ほど",
                   "action": "火の中で刃を一度ひっくり返す",
                   "reason": "峰の方は火の当たりが遅れる。返さないと峰だけ焼きが甘くなる。"},
            "en": {"title": "Turn the blade once during the soak",
                   "trigger": "About halfway through the soak, once the colour is even",
                   "action": "Roll the blade over once in the fire",
                   "reason": "The spine lags behind in the fire. Skip the turn and the spine comes out soft."},
            "vi": {"title": "Lật lưỡi dao một lần khi ủ nhiệt",
                   "trigger": "Khoảng giữa thời gian ủ, khi màu đã đều",
                   "action": "Lật lưỡi dao một lần trong lửa",
                   "reason": "Sống dao nóng chậm hơn. Không lật thì sống dao sẽ bị mềm."},
            "id": {"title": "Balik bilah sekali saat penahanan panas",
                   "trigger": "Sekitar pertengahan waktu tahan, saat warnanya sudah rata",
                   "action": "Balik bilah sekali di dalam api",
                   "reason": "Punggung bilah lebih lambat panas. Tanpa dibalik, punggungnya jadi lunak."},
        },
    },
    "check": {
        "transcript": "入れる直前に一瞬火から出して、影で色を見る。火の中だと明るく見えすぎる。外で桜色なら水に入れていい。",
        "content": {
            "ja": {"title": "入れる前に火の外で色を見る",
                   "trigger": "焼入れの直前",
                   "action": "一瞬だけ火から出し、影で色を確かめる。外で桜色なら焼きを入れる。",
                   "reason": "火の中では照り返しで色が明るく見えすぎる。外で見ないと温度を読み違える。"},
            "en": {"title": "Read the colour outside the fire before quenching",
                   "trigger": "Just before the quench",
                   "action": "Pull the blade out for a second and read the colour in shadow. Cherry outside means quench now.",
                   "reason": "Inside the fire the glare makes steel look hotter than it is. Read it there and you misjudge the heat."},
            "vi": {"title": "Xem màu bên ngoài lửa trước khi tôi",
                   "trigger": "Ngay trước khi tôi",
                   "action": "Rút lưỡi ra một giây và xem màu trong bóng râm. Màu anh đào bên ngoài nghĩa là tôi ngay.",
                   "reason": "Trong lửa, ánh sáng làm thép trông nóng hơn thực tế. Xem trong lửa sẽ đoán sai nhiệt độ."},
            "id": {"title": "Baca warna di luar api sebelum dicelup",
                   "trigger": "Tepat sebelum pencelupan",
                   "action": "Tarik bilah keluar sedetik dan lihat warnanya di tempat teduh. Merah ceri di luar berarti celup sekarang.",
                   "reason": "Di dalam api, silau membuat baja tampak lebih panas dari sebenarnya. Membaca di sana membuat salah menilai suhu."},
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

    _seed_heat(service, now, day)

    real = service.create_session("kaiten", "grinder", source="video:media/grinder.webm", created_at=now - 2 * day)
    service.finalize(real["id"], np.load(FOOTAGE))

    for person, profile, sd, days in APPRENTICE_RUNS:
        ses = service.create_session("honbazuke", person, source="seed", created_at=now - days * day)
        frames, _ = generate(profile, seed=sd)
        service.finalize(ses["id"], frames)
        for cid in card_ids:
            store.log_card_view(cid, person, ses["id"], at=now - days * day)

    return {"seeded": True, "master_session": master["id"]}


def _seed_heat(service: Tsukumo, now: float, day: float) -> None:
    """Yamada's quench-hardening session, his two cards, and two apprentices trying it."""
    store = service.store
    master = service.create_session("yakiire", "yamada", source="seed", created_at=now - 10 * day)
    frames, events = generate_heat("master", seed=0)
    service.finalize(master["id"], frames)
    card_ids, used = [], set()
    for s in store.surprises(master["id"]):
        habit = _match([s], events).get(s["id"])
        if habit in HEAT_CARDS and habit not in used:
            used.add(habit)
            spec = HEAT_CARDS[habit]
            card = store.add_card(skill_id="yakiire", master_id="yamada", stroke_index=s["stroke_index"],
                                  content=spec["content"], source="seed", session_id=master["id"],
                                  surprise_id=s["id"], transcript=spec["transcript"], transcript_lang="ja",
                                  evidence={"kind": s["kind"], "summary": s["summary"], "t_start": s["t_start"]})
            store.set_surprise_status(master["id"], s["id"], "answered", card["id"])
            card_ids.append(card["id"])
    for person, profile, sd, days in (("nguyen", "novice", 41, 4), ("sato", "skilled", 42, 1)):
        ses = service.create_session("yakiire", person, source="seed", created_at=now - days * day)
        service.finalize(ses["id"], generate_heat(profile, seed=sd)[0])
        for cid in card_ids:
            store.log_card_view(cid, person, ses["id"], at=now - days * day)
