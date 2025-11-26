from pathlib import Path
from src.dataio.report_generator import generate_report

labels_csv = Path("data/labels/labels_gt.csv")
preds_dir  = Path("outputs/preds_json/")   # carpeta con JSON de predicción
out_xlsx   = Path("reports/reporte_8yang.xlsx")

generate_report(labels_csv, preds_dir, out_xlsx, alias="modelo_rf_v1")
