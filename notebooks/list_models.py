"""ขั้นที่ 4: ดูรายชื่อโมเดล Longevity-LLM และขนาดไฟล์ — ยังไม่โหลดอะไรลงเครื่อง"""
from huggingface_hub import HfApi

api = HfApi()
models = list(api.list_models(author="insilicomedicine", full=True))
models += [m for m in api.list_models(search="Longevity", author="LiquidAI", full=True)]

for m in models:
    try:
        info = api.model_info(m.id, files_metadata=True)
        size = sum((s.size or 0) for s in info.siblings) / 1e9
    except Exception as e:
        size = float("nan")
    print(f"{m.id:60} {size:6.2f} GB")
