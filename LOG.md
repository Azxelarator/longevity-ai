# Longevity AI — Work Log

## 2026-09-25
- ทำอะไร: ตั้งโปรเจกต์ (venv, git, โครงโฟลเดอร์, README, explore.py)
- ได้ผลอะไร: โครงพร้อม รอรัน explore.py
- ติดอะไร / ครั้งหน้าจะทำอะไร: รัน explore.py ดูรายชื่อ config ของ longebench แล้วเลือกมาเปิดดูหนึ่งชุด
- อัปเดต: explore.py ผ่าน — longebench มี 3 config: benchmark, mini, extra
- เพิ่ม run.bat (เปิด venv อัตโนมัติ) และ notebooks/peek.py (streaming ดูชุด mini)
- peek.py ผ่าน: mini มี split 'eval'; แต่ละแถว = 1 ข้อสอบ (messages = คำถาม, metadata.follow_up = เฉลย)
  ตัวอย่าง: nhanes_age_pairwise — ดูผลเลือด 2 คน ตอบว่าใครแก่กว่า (A/B)
- เพิ่ม notebooks/summary.py (นับจำนวนโจทย์แต่ละแบบ)
- summary.py ผ่าน: mini = 728 ข้อ, 17 task (clinical/epigenomics/genetics/proteomics/transcriptomics)
  format: binary, pairwise, multiclass (accuracy) / regression (mae)
- เป้าถัดไป (Mk.1): รัน Longevity-LLM ตัวเล็กบนเครื่อง แล้ววัดคะแนนเองบน nhanes_age_pairwise
- เพิ่ม notebooks/list_models.py
- list_models: เลือก LiquidAI/LFM2-1.2B-Longevity (2.35 GB) เป็นตัวหลัก
- เพิ่ม notebooks/eval_pairwise.py — วัด accuracy เองบน nhanes_age_pairwise
- fix: transformers ใหม่ apply_chat_template คืน dict -> ใช้ return_dict=True + generate(**enc); torch_dtype->dtype

## 2026-09-25 (ต่อ) — Mk.1 ผลแรก
- LFM2-1.2B-Longevity, nhanes_age_pairwise (mini, 45 ข้อ): 27/45 = 60.0% | 15.5s | VRAM 2.59 GB
- ข้อสังเกต: เอียงไปทาง B — gold=B ถูก 16/23, gold=A ถูก 11/22
- 45 ข้อน้อยเกินจะสรุปว่าดีกว่าเดามั่ว
- ถัดไป (Mk.2): notebooks/eval_swap.py — benchmark 200 ข้อ + ทดสอบสลับ A<->B

## 2026-09-25 (ต่อ) — Mk.2 ทดสอบสลับ A<->B
- benchmark 200 ข้อ nhanes_age_pairwise | 133s
- acc ปกติ 69.5% | acc สลับ 77.5% | consistent 65.0% | ถูกทั้งสองรอบ 56.0% (สุ่ม ~25%)
- ตอบ A 134 / B 266 -> เอียง B หนัก (~1/3 ของข้อ ไม่สลับตามข้อมูล)
- สรุป: โมเดลอ่านผลเลือดได้จริง แต่มี letter bias ที่เปเปอร์ไม่ได้รายงาน
- ถัดไป Mk.2.1: notebooks/analyze_swap.py — ใช้ consistency เป็นตัวกรองความมั่นใจ

## 2026-09-25 (ต่อ) — Mk.2.1 consistency filter (ไม่ต้องเทรน)
- ถามรอบเดียว: 69.5% (ตอบ 100%) -> เชื่อเฉพาะ consistent: 86.2% (ตอบ 65%, 130/200)
- ข้อไม่แน่ใจ 70 ข้อ: B/B 68, A/A 2 -> ยืนยันว่าเป็น letter bias
- ตามช่วงอายุห่าง: 0-10 ปี 68.4% | 10-20 71.0% | 20-40 94.3% | 40+ 100%
- โจทย์จริงที่ต้องแก้: แยกคนอายุใกล้กัน (<20 ปี)
- คำถามถัดไป: bias นี้เกิดกับ domain อื่นด้วยไหม (olink/gtex/geodnam pairwise)

## 2026-09-25 (ต่อ) — Mk.3 เตรียมทดสอบทุก domain
- peek_pairwise: รูปแบบต่างกันมาก — nhanes (Participant A/B เป็นก้อน), geodnam/olink (A:/B: ปนทุกบรรทัด),
  gtex (Sample a / Sample-A), nhanes_mortality (เฉลยมี censoring — ข้ามก่อน), synergyage (ไม่มีเฉลย — ตัด)
- เปลี่ยนวิธีสลับ: จากย้ายก้อน -> สลับ "ป้าย" ทุกตำแหน่ง (ระวัง "A male" = คำนำหน้า ไม่ใช่ป้าย)
- notebooks/eval_swap_all.py มีโหมด check ตรวจการสลับก่อนรันโมเดล
- check mode ผ่านทั้ง 4 ชุด (geodnam ป้าย B มากกว่า A ตั้งแต่ต้น 306:176)
- Mk.3 รอบแรก OOM: geodnam ขอ 68 GiB (math attention สร้างตาราง LxL) -> บังคับ flash/efficient SDPA, ส่งแค่ input_ids, ข้ามข้อที่ OOM
- รอบ 2: No available kernel — GQA (q 32 หัว / kv 8 หัว) ใช้ efficient kernel ไม่ได้ -> ปิด use_gqa_in_sdpa ให้ repeat_kv ก่อน

## 2026-09-25 (ต่อ) — Mk.3 ผล label swap 4 domain (n=30/ชุด)
- acc -> acc_sw: nhanes 70->47 | geodnam 80->33 | gtex 67->40 | olink 47->33
- consistency: 50 / 33 / 7 / 17 %
- เทียบกับ block swap (nhanes 200): acc_sw 77.5% -> สรุปชั่วคราว: โมเดลผูกคำตอบกับ "ตำแหน่ง" ไม่ใช่ "ป้าย"
- ข้อระวัง: label swap ทำให้ B มาก่อน A (นอกรูปแบบที่เทรน) + n=30 แกว่ง ±15%
- geodnam เฉลี่ย 22k tokens, 573s — ช้าสุด; VRAM สูงสุด 4.26 GB
- เพิ่ม ROADMAP.md (5 ด่านของตาชั่ง + ด่านหลังจากนั้น)
- ถัดไป Mk.4: order-invariant scoring (เฉลี่ยความน่าจะเป็นจากทั้งสองลำดับ)

## 2026-09-25 (ต่อ) — Mk.4 order-invariant scoring ✅ ผ่านขั้นที่ 2 (nhanes)
- nhanes_age_pairwise, benchmark 200 ข้อ, 102s
- ถามลำดับเดียว 70.5% -> เฉลี่ยความน่าจะเป็น 2 ลำดับ 79.0% | consistency 100% โดยนิยาม | ตอบครบ 100% ของข้อ
- เทียบ Mk.2.1 (consistency filter): 86.2% แต่ตอบแค่ 65% -> Mk.4 ดีกว่าเพราะไม่ต้องทิ้งข้อ
- bias เดิม: mean P(A) = 0.39 (ควร 0.50) -> หลังแก้ ทาย A/B = 92/108
- ตามช่วงอายุห่าง: 0-10 62.9% | 10-20 67.3% | 20-40 85.4% | 40+ 100%
- จุดอ่อนที่เหลือ: แยกคนอายุต่างกัน < 20 ปี
- ถัดไป: ขั้นที่ 3 — baseline แบบง่าย (regression บนค่าเลือดตรงๆ) เทียบบนข้อสอบเดียวกัน
- เพิ่ม README.md (อังกฤษ, สรุปผล Mk.1–4), LICENSE (MIT), ย้าย README เดิมเป็น README.th.md — เตรียมขึ้น GitHub

## 2026-09-25 (ต่อ) — เตรียมตรวจความถูกต้อง
- leaderboard (longevitybenchmarks.org) ไม่มีคะแนนแยกราย task ให้เทียบ -> ขุดจาก README ชุดข้อมูลแทน
- เพิ่ม notebooks/verify.py: นับทั้งชุด, สุ่มตรวจเฉลย, เทียบ 200 ข้อแรก vs ที่เหลือ, หา split/คะแนนใน README
- checklist ก่อนให้คนอื่นใช้: ตรวจเฉลย / ใช้ทั้งชุด / ตรวจ leakage / เทียบเปเปอร์ / ใส่ช่วงคลาดเคลื่อน / ล็อกเวอร์ชัน + commit ผล
- verify.py ผ่าน: nhanes_age_pairwise benchmark = 2102 ข้อ, ไม่มีอายุเท่ากัน, A 1075 / B 1027
  สุ่ม 5 ข้อ เฉลยถูกทั้งหมด | อายุห่างเฉลี่ย 200 แรก 24.9 vs ที่เหลือ 24.4 ปี -> 200 แรกเป็นตัวแทนได้
  README: มีแค่ split eval, ไม่มีคะแนนแยก task ให้เทียบ; leakage ตรวจเองไม่ได้ (เปเปอร์บอกแบ่ง NHANES ตาม survey wave)
- eval_invariant.py รองรับ "all" + ช่วงเชื่อมั่น 95% (Wilson)

## 2026-09-25 (ต่อ) — Mk.4 ทั้งชุด ✅ ยืนยันแล้ว
- nhanes_age_pairwise benchmark ทั้งหมด 2102 ข้อ, 645s
- ลำดับเดียว 71.2% [95% CI 69.2–73.1] -> order-invariant 79.2% [77.4–80.8] — ช่วงไม่ทับกัน = ดีกว่าจริง ไม่ใช่ฟลุ๊ค
- mean P(A) 0.41 -> หลังแก้ ทาย A/B 1043/1059 (เฉลยจริง 1075/1027)
- ตามช่วงอายุห่าง: 0-10 57.7% (326) | 10-20 71.2% (607) | 20-40 86.2% (796) | 40+ 96.0% (373)
- ผล 200 ข้อ (79.0%) ใกล้กับทั้งชุด (79.2%) -> ยืนยันว่า 200 ข้อแรกเป็นตัวแทนได้
