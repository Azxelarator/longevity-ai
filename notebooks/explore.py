"""ขั้นที่ 1: ดูว่าใน longebench มีชุดย่อย (config) อะไรบ้าง — ไม่โหลดข้อมูลลงดิสก์"""
from datasets import get_dataset_config_names

NAME = "insilicomedicine/longebench"

configs = get_dataset_config_names(NAME)
print(f"พบ {len(configs)} ชุดย่อย:")
for c in configs:
    print(" -", c)
