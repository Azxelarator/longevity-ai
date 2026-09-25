"""ขั้นที่ 2: เปิดดูชุด 'mini' แบบ streaming — ไม่โหลดทั้งก้อนลงดิสก์
ดูว่ามีคอลัมน์อะไร และตัวอย่างข้อมูล 2 แถวแรกหน้าตาเป็นยังไง"""
from datasets import load_dataset

NAME = "insilicomedicine/longebench"
CONFIG = "mini"

ds = load_dataset(NAME, CONFIG, streaming=True)
print("splits:", list(ds.keys()))

split = list(ds.keys())[0]
for i, row in enumerate(ds[split]):
    print(f"\n===== แถวที่ {i+1} ({split}) =====")
    for k, v in row.items():
        s = str(v)
        print(f"[{k}] {s[:500]}{' ...(ตัด)' if len(s) > 500 else ''}")
    if i >= 1:
        break
