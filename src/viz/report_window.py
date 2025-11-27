# src/viz/report_window.py
from __future__ import annotations

from typing import Optional
import os

import pandas as pd
from PyQt5 import QtWidgets, QtCore


class ReportWindow(QtWidgets.QDialog):
    """Diálogo para inspeccionar rápidamente un archivo de reporte.

    Esta ventana carga una hoja Excel en memoria y la muestra en una
    tabla no editable. Es una herramienta de visualización enfocada en
    facilitar la revisión rápida de resultados durante el desarrollo.
    """
    def __init__(self, xlsx_path: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.setWindowTitle(f"Reporte Poomsae - {os.path.basename(xlsx_path)}")
        self.resize(800, 400)

        layout = QtWidgets.QVBoxLayout(self)

        # Cargar DataFrame (intenta leer la primera hoja del Excel)
        try:
            df = pd.read_excel(xlsx_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error al leer reporte",
                f"No se pudo leer el archivo:\n{xlsx_path}\n\n{e}",
            )
            self.close()
            return

        table = QtWidgets.QTableWidget(self)
        table.setRowCount(df.shape[0])
        table.setColumnCount(df.shape[1])
        table.setHorizontalHeaderLabels(df.columns.astype(str).tolist())
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                val = df.iat[r, c]
                item = QtWidgets.QTableWidgetItem(str(val))
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                table.setItem(r, c, item)

        layout.addWidget(table)

        btn_close = QtWidgets.QPushButton("Cerrar", self)
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)
