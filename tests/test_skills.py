"""
Tests para skills y servidor MCP.
"""

import pytest


def test_report_skill_daily() -> None:
    """Verifica que el reporte diario contiene la palabra 'Reporte'."""
    try:
        from src.skills.report_skill import generate_sales_report

        result = generate_sales_report("diario")

        assert "report" in result
        assert "summary" in result
        assert "REPORTE" in result["report"] or "Reporte" in result["report"]
        assert result["summary"]["period"].lower() == "diario"

        assert "Ventas" in result["report"] or "ventas" in result["report"]
        assert "Total de ventas" in result["report"]
        assert "Facturación" in result["report"]
        assert "TOP 5" in result["report"]

    except ImportError as e:
        pytest.skip(f"Requiere dependencias de skills: {e}")


def test_report_skill_weekly() -> None:
    """Verifica que el reporte semanal tiene estructura correcta."""
    try:
        from src.skills.report_skill import generate_sales_report

        result = generate_sales_report("semanal")

        assert result["summary"]["period"].lower() == "semanal"
        assert result["summary"]["total_sales"] > result["summary"]["top_products"][0]["units"]

    except ImportError as e:
        pytest.skip(f"Requiere dependencias de skills: {e}")


def test_report_skill_monthly() -> None:
    """Verifica que el reporte mensual tiene datos completos."""
    try:
        from src.skills.report_skill import generate_sales_report

        result = generate_sales_report("mensual")

        assert result["summary"]["period"].lower() == "mensual"
        assert len(result["summary"]["top_products"]) == 5
        assert result["summary"]["top_products"][0]["revenue"] > 0

    except ImportError as e:
        pytest.skip(f"Requiere dependencias de skills: {e}")


def test_mcp_get_stock() -> None:
    """Verifica que el servidor MCP retorna stock de un producto."""
    try:
        from src.mcp.inventory_mcp_server import get_product_stock

        result = get_product_stock("camisa blanca")

        assert result["found"] is True
        assert "Camisa Blanca" in result["product"]
        assert result["stock"] > 0
        assert result["price"] > 0
        assert "status" in result

    except ImportError as e:
        pytest.skip(f"Requiere dependencias de MCP: {e}")


def test_mcp_product_not_found() -> None:
    """Verifica que el servidor MCP maneja productos no encontrados."""
    try:
        from src.mcp.inventory_mcp_server import get_product_stock

        result = get_product_stock("producto_inexistente_xyz")

        assert result["found"] is False
        assert "message" in result

    except ImportError as e:
        pytest.skip(f"Requiere dependencias de MCP: {e}")


def test_mcp_low_stock_alerts() -> None:
    """Verifica que las alertas de stock bajo retornan productos críticos."""
    try:
        from src.mcp.inventory_mcp_server import get_low_stock_alerts

        alerts = get_low_stock_alerts(threshold=10)

        assert isinstance(alerts, list)
        if alerts:
            assert "product" in alerts[0]
            assert "current_stock" in alerts[0]
            assert "suggested_order" in alerts[0]
            assert alerts[0]["current_stock"] < 10 or alerts[0]["current_stock"] < alerts[0]["min_stock"]

    except ImportError as e:
        pytest.skip(f"Requiere dependencias de MCP: {e}")


def test_mcp_sales_summary() -> None:
    """Verifica que el resumen de ventas tiene estructura correcta."""
    try:
        from src.mcp.inventory_mcp_server import get_sales_summary

        summary = get_sales_summary("daily")

        assert "total_sales" in summary
        assert "total_revenue" in summary
        assert "average_ticket" in summary
        assert "top_products" in summary
        assert len(summary["top_products"]) == 5
        assert summary["low_stock_alerts"] >= 0

    except ImportError as e:
        pytest.skip(f"Requiere dependencias de MCP: {e}")
