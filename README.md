# Asistente IA – Gestión de Inventario con RAG y Multiagentes

## Descripción

Sistema de asistente inteligente especializado en **gestión de inventario y ventas para tiendas de ropa**, inspirado en el sistema "Keep Control" para el almacén K Bonita. El asistente responde preguntas en lenguaje natural sobre inventario, ventas, devoluciones, reportes y gestión de productos, utilizando **RAG** (Retrieval-Augmented Generation) sobre un corpus documental, coordinando **dos agentes** con roles distintos mediante **LangGraph**, e integrando **MCP** y **Skills**.

## Arquitectura

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Streamlit UI   │────▶│  LangGraph       │────▶│  RetrieverAgent │
│  (Interfaz)     │     │  (Orquestación)  │     │  (Búsqueda)     │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
         │                       │                         │
         │                       │                         ▼
         │                       │              ┌─────────────────┐
         │                       │              │  ChromaDB       │
         │                       │              │  (Vector Store) │
         │                       │              └─────────────────┘
         │                       │                         │
         ▼                       ▼                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  ReportSkill    │     │  WriterAgent     │     │  Embeddings     │
│  (Reportes)     │     │  (Redacción)     │     │  all-MiniLM     │
└────────┬────────┘     └────────┬─────────┘     └─────────────────┘
         │                       │                         │
         ▼                       ▼                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  MCP Server     │     │  LLM             │     │  Corpus         │
│  Inventario     │     │  Groq/Llama 3.2  │     │  15 documentos  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Componentes

- **Interfaz Streamlit**: Aplicación web interactiva para consultar el asistente y generar reportes.
- **Orquestador LangGraph**: Coordina el flujo de trabajo entre el agente recuperador y el agente redactor.
- **Agente Recuperador**: Analiza la pregunta, reformula queries y busca en la base vectorial ChromaDB.
- **Agente Redactor**: Recibe los documentos recuperados y redacta la respuesta final con citación de fuentes.
- **ChromaDB**: Base de datos vectorial persistente que almacena los embeddings de los documentos.
- **Pipeline RAG**: Búsqueda híbrida (semántica + BM25) con re-ranking por relevancia.
- **Servidor MCP**: Expone herramientas de consulta de inventario simulado (stock, alertas, ventas).
- **Skill de Reportes**: Genera reportes estructurados de ventas consumiendo el servidor MCP.

## Instalación

### 1. Clonar el repositorio

```bash
git clone [url-del-repositorio]
cd asistente_ia_inventario
```

### 2. Crear entorno virtual

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

```bash
cp .env.example .env
```

Edita el archivo `.env` y completa al menos una API key:
- **Groq**: `GROQ_API_KEY=tu_api_key` 

### 5. Indexar documentos en la base vectorial

```bash
python src/rag/ingestion.py
```

Este comando carga los 15 documentos de `data/raw/`, genera chunks y embeddings, y los almacena en ChromaDB.

### 6. Ejecutar la aplicación

```bash
streamlit run src/ui/app.py
```

## Ejemplos de preguntas

- "¿Cómo se manejan las devoluciones en el sistema?"
- "¿Qué métodos de control de inventario existen?"
- "Explica el proceso de facturación electrónica según la DIAN"
- "¿Cuáles son los KPIs más importantes para una tienda de ropa?"
- "¿Qué requisitos funcionales tiene el módulo de ventas?"
- "¿Cómo se gestionan los usuarios y roles?"
- "Genera un reporte de ventas semanal"

## Estructura del proyecto

```
asistente_ia_inventario/
├── src/
│   ├── agents/           # Agentes Recuperador y Redactor
│   ├── rag/              # Pipeline RAG completo
│   ├── mcp/              # Servidor MCP de inventario
│   ├── skills/           # Skill de generación de reportes
│   ├── graph/            # Orquestación con LangGraph
│   └── ui/               # Interfaz Streamlit
├── data/
│   ├── raw/              # Corpus documental (15 archivos)
│   └── chroma_db/        # Base vectorial persistente
├── docs/                 # Documentación y diagramas
├── tests/                # Tests unitarios
├── .env.example          # Ejemplo de configuración
├── requirements.txt      # Dependencias
└── README.md             # Este archivo
```

## Proyectos relacionados
- Backend Keep Control: https://github.com/StivenVelez2/backendkeepcontrol
- Frontend Keep Control: https://github.com/StivenVelez2/frontedkeepcontrol

## Integrantes

- Johan Stiven Velez
- Ryan Peña Ramirez

## URL del repositorio

https://github.com/StivenVelez2/StivenVelez2-asistente-ia-inventario
