import asyncio
from app.services.coco_export import COCOExporter

def run():
    exporter = COCOExporter()
    res = exporter.export()
    print("Exported to:", res["path"])

if __name__ == "__main__":
    run()
