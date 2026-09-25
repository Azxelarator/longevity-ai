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
