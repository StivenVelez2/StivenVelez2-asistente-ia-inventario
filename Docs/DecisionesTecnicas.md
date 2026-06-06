# Decisiones Técnicas del Proyecto

## Decisión 1: ¿Por qué usamos ChromaDB para guardar los documentos?

**El problema**: Necesitábamos un lugar donde guardar los documentos del sistema de forma que la IA pudiera buscar en ellos rápidamente. No cualquier base de datos sirve para esto  necesitábamos una que entienda el significado de las palabras, no solo el texto exacto.

**Lo que elegimos**: ChromaDB, una base de datos especial para inteligencia artificial que guarda los documentos en el disco duro del computador.

**Por qué no elegimos otras opciones**:
- Pinecone: necesita internet y tiene límites en el plan gratuito
- FAISS: no guarda los datos automáticamente cuando se apaga el computador
- Weaviate: requiere instalar software adicional complejo (Docker)

**Resultado**: El sistema funciona completamente sin internet, los documentos quedan guardados permanentemente y es fácil de configurar. La desventaja es que no escalaría bien si tuviéramos millones de documentos, pero para este proyecto es más que suficiente.

---

## Decisión 2: ¿Por qué combinamos dos tipos de búsqueda?

**El problema**: Cuando alguien le pregunta algo al asistente, este tiene que buscar en los documentos la información relevante. Hay dos formas de buscar:
- **Búsqueda por significado**: entiende que "devolución" y "reembolso" son cosas similares
- **Búsqueda por palabras exactas**: útil para términos técnicos como "RF01", "SKU" o "DIAN"

Ninguna de las dos por sí sola es perfecta.

**Lo que elegimos**: Combinar las dos búsquedas al mismo tiempo. El 70% del peso lo tiene la búsqueda por significado y el 30% la búsqueda por palabras exactas. Esta combinación se llama "búsqueda híbrida" y es una mejora sobre el RAG básico.

**Por qué no elegimos solo una**:
- Solo por significado: falla cuando el usuario escribe códigos o términos muy específicos
- Solo por palabras exactas: no entiende sinónimos ni contexto

**Resultado**: El asistente encuentra información relevante en más casos. Si preguntas por "RF01" lo encuentra exactamente, y si preguntas "¿cómo registro una venta?" también entiende el contexto aunque no use esas palabras exactas.

---

## Decisión 3: ¿Por qué usamos LangGraph para coordinar los agentes?

**El problema**: El sistema tiene dos agentes que trabajan juntos — uno busca la información y otro redacta la respuesta. Necesitábamos una forma de coordinarlos, pasarles información entre ellos y controlar qué pasa si algo falla.

**Lo que elegimos**: LangGraph, una herramienta que permite definir el flujo de trabajo entre agentes como si fuera un mapa: primero va el Agente Recuperador, luego el Agente Redactor, y si no se encontró información el sistema responde que no tiene datos suficientes.

**Por qué no elegimos otras opciones**:
- CrewAI: es más complejo y oculta demasiado lo que pasa por dentro, haciendo difícil explicarlo en la sustentación
- Secuencia manual: simplemente llamar una función tras otra sin estructura formal, difícil de mantener y escalar

**Resultado**: Cada agente hace exactamente lo que le corresponde, el flujo es claro y fácil de explicar, y si en el futuro se quisiera agregar un tercer agente (por ejemplo uno que valide la respuesta) se puede hacer sin cambiar todo el código.