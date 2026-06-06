import logging
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BACKEND_URL = "http://localhost:8000"


def _fetch_json(url: str) -> dict[str, Any] | list[Any]:
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        logger.error("Error consultando %s: %s", url, e)
        return {"error": str(e)}


def generate_sales_report(period: str = "daily") -> dict[str, Any]:
    dashboard = _fetch_json(f"{BACKEND_URL}/dashboard")
    ventas = _fetch_json(f"{BACKEND_URL}/ventas")

    if isinstance(dashboard, dict) and "error" in dashboard:
        return {"report": "Error: No se pudo conectar con el backend.", "summary": {}}

    total_ventas = dashboard.get("total_ventas", 0) if isinstance(dashboard, dict) else 0
    ingresos = dashboard.get("ingresos_totales", 0) if isinstance(dashboard, dict) else 0
    total_devoluciones = dashboard.get("total_devoluciones", 0) if isinstance(dashboard, dict) else 0
    productos_top = (
        dashboard.get("productos_mas_vendidos") or []
        if isinstance(dashboard, dict)
        else []
    )
    ventas_list = ventas if isinstance(ventas, list) else []

    avg_ticket = round(ingresos / total_ventas, 2) if total_ventas > 0 else 0

    today = datetime.now()
    report_lines = []

    report_lines.append("=" * 60)
    report_lines.append(f"  REPORTE DE VENTAS - KEEP CONTROL")
    report_lines.append(f"  Período: {period.upper()}")
    report_lines.append(f"  Fecha de generación: {today.strftime('%d/%m/%Y %H:%M')}")
    report_lines.append("=" * 60)
    report_lines.append("")

    report_lines.append("── RESUMEN DE VENTAS ──")
    report_lines.append(f"  Total de ventas:       {total_ventas}")
    report_lines.append(f"  Facturación total:     ${ingresos:,.0f}")
    report_lines.append(f"  Ticket promedio:       ${avg_ticket:,.0f}")
    report_lines.append(f"  Devoluciones:          {total_devoluciones}")
    report_lines.append("")

    report_lines.append("── TOP 5 PRODUCTOS MÁS VENDIDOS ──")
    report_lines.append(f"  {'Producto':<35} {'Unds':<8}")
    report_lines.append("  " + "-" * 43)
    for i, prod in enumerate(productos_top[:5], 1):
        report_lines.append(
            f"  {i}. {prod.get('nombre', '')[:33]:<33} {prod.get('total_vendido', 0):<8}"
        )
    report_lines.append("")

    report_lines.append("── RECOMENDACIONES AUTOMÁTICAS ──")

    if total_ventas < 10:
        report_lines.append("  - El volumen de ventas es bajo. Revisar estrategias de promoción.")
    elif total_ventas > 50:
        report_lines.append("  - Buen volumen de ventas. Asegurar reposición oportuna de productos top.")
    else:
        report_lines.append("  - Volumen de ventas normal. Continuar con la operación estándar.")

    if avg_ticket < 100000:
        report_lines.append("  - Ticket promedio bajo. Considerar estrategias de venta cruzada y upsell.")
    elif avg_ticket > 250000:
        report_lines.append("  - Ticket promedio alto. Mantener enfoque en productos premium.")

    if total_devoluciones > 5:
        report_lines.append(f"  - Alta tasa de devoluciones ({total_devoluciones}). Revisar calidad de productos.")

    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("  Generado por Keep Control - Sistema de Gestión de Inventario")
    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)

    return {
        "report": report_text,
        "summary": {
            "period": period,
            "total_sales": total_ventas,
            "total_revenue": ingresos,
            "average_ticket": avg_ticket,
            "top_products": productos_top,
            "low_stock_count": 0,
            "generated_at": today.isoformat(),
        },
    }
