"""Paquete `dataio`.

Este __init__ expone funciones útiles desde `report_generator`, pero NO debe
ejecutar lógica al importarse. Mantener la inicialización de generación de
reportes fuera del import evita errores al importar el paquete desde otros
módulos (p. ej. durante la inicialización de la UI).
"""

from .report_generator import generate_pal_yang_report

__all__ = ["generate_pal_yang_report"]
