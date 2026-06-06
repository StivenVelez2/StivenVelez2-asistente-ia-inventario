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


def get_product_stock(product_name: str) -> dict[str, Any]:
    data = _fetch_json(f"{BACKEND_URL}/productos")
    if isinstance(data, dict) and "error" in data:
        return {"found": False, "product": product_name, "message": data["error"]}

    productos = data if isinstance(data, list) else []
    search = product_name.lower().replace(" ", "_")

    for prod in productos:
        nombre = prod.get("nombre", "")
        if search in nombre.lower().replace(" ", "_") or product_name.lower() in nombre.lower():
            stock = prod.get("stock", 0)
            return {
                "found": True,
                "product": nombre,
                "category": prod.get("categoria", ""),
                "stock": stock,
                "price": prod.get("precio", 0),
                "status": "Bajo stock" if stock < 10 else "Disponible",
            }

    return {"found": False, "product": product_name, "message": "Producto no encontrado"}


def get_low_stock_alerts(threshold: int = 10) -> list[dict[str, Any]]:
    data = _fetch_json(f"{BACKEND_URL}/productos")
    if isinstance(data, dict) and "error" in data:
        return [{"error": data["error"]}]

    productos = data if isinstance(data, list) else []
    alerts = []

    for prod in productos:
        stock = prod.get("stock", 0)
        if stock < threshold:
            alerts.append({
                "product": prod.get("nombre", ""),
                "category": prod.get("categoria", ""),
                "current_stock": stock,
                "min_stock": threshold,
                "suggested_order": max(threshold * 2 - stock, 0),
            })

    alerts.sort(key=lambda x: x["current_stock"])
    return alerts


def get_sales_summary(period: str = "daily") -> dict[str, Any]:
    data = _fetch_json(f"{BACKEND_URL}/dashboard")
    if isinstance(data, dict) and "error" in data:
        return {"error": data["error"]}

    today = datetime.now()
    total_ventas = data.get("total_ventas", 0)
    ingresos = data.get("ingresos_totales", 0)
    avg_ticket = round(ingresos / total_ventas, 2) if total_ventas > 0 else 0

    top_products = [
        {"name": p.get("nombre", ""), "units": p.get("total_vendido", 0), "revenue": 0}
        for p in (data.get("productos_mas_vendidos") or [])
    ]

    summary = {
        "period": period.lower(),
        "date": today.strftime("%Y-%m-%d"),
        "total_sales": total_ventas,
        "total_revenue": ingresos,
        "average_ticket": avg_ticket,
        "top_products": top_products,
        "payment_methods": {},
    }
    summary["low_stock_alerts"] = len(get_low_stock_alerts(10))
    return summary


def handle_mcp_request(request: dict[str, Any]) -> dict[str, Any]:
    tool = request.get("tool", "")
    params = request.get("params", {})

    if tool == "get_product_stock":
        result = get_product_stock(params.get("product_name", ""))
        return {"success": True, "result": result}

    elif tool == "get_low_stock_alerts":
        result = get_low_stock_alerts(params.get("threshold", 10))
        return {"success": True, "result": result}

    elif tool == "get_sales_summary":
        result = get_sales_summary(params.get("period", "daily"))
        return {"success": True, "result": result}

    else:
        return {
            "success": False,
            "error": f"Herramienta desconocida: {tool}",
        }
