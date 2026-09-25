# Longevity AI

เป้าหมาย: สร้าง AI ที่ช่วยตีความข้อมูลชีวภาพเรื่องความแก่ (aging) — ไม่แสวงกำไร เปิดให้คนอื่นใช้ต่อ

## หลักการ
- ไม่รับประกันความสำเร็จ แต่บอกความจริงตลอดทาง
- ทรัพยากร: แล็ปท็อปเครื่องเดียว (i5-13420H, RAM 16GB, RTX 4050 6GB) — รีดทุกไบต์
- งานหนักส่งไป Kaggle/Colab (GPU ฟรี)

## จุดเริ่มต้น
- Dataset: [insilicomedicine/longebench](https://huggingface.co/datasets/insilicomedicine/longebench) (CC-BY-NC-4.0)
- Models: [Longevity-LLM collection](https://huggingface.co/collections/insilicomedicine/longevity-llm)
- Paper: Zhavoronkov et al., Cell 2026, doi:10.1016/j.cell.2026.08.026

## โครงสร้าง
```
notebooks/   ไฟล์ทดลอง สำรวจข้อมูล
src/         โค้ดที่ใช้ซ้ำได้
results/     ผลลัพธ์ (ไม่ขึ้น git)
LOG.md       สมุดบันทึกทุกครั้งที่ทำงาน
```

## เริ่มทำงานทุกครั้ง (Windows cmd)
```
cd C:\longevity
.venv\Scripts\activate
```
