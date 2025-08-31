# Documentación del Sistema RAG de UDI

## **Descripción General**

El Sistema RAG (Retrieval-Augmented Generation) de UDI es una arquitectura avanzada de procesamiento de consultas que combina **recuperación de información inteligente** con **generación de respuestas contextuales**. Está diseñado específicamente para manejar consultas sobre documentación universitaria de UDIT, proporcionando respuestas precisas y contextualmente relevantes.

## **Arquitectura del Sistema**

### **Componentes Principales**

#### 1. **RAGSystem** (`src/rag/rag_system.py`)
- **Función**: Orquestador principal del sistema RAG
- **Responsabilidades**:
  - Coordinación entre todos los componentes
  - Clasificación inteligente de consultas
  - Selección del método de respuesta (RAG vs GPT)
  - Gestión del flujo de procesamiento

#### 2. **VectorStore** (`src/rag/vector_store.py`)
- **Función**: Almacenamiento y búsqueda de embeddings vectoriales
- **Tecnología**: FAISS (Facebook AI Similarity Search)
- **Características**:
  - Modelo de embeddings: `sentence-transformers/all-MiniLM-L6-v2`
  - Búsqueda por similitud semántica
  - Cache persistente en disco
  - Búsqueda por categorías

#### 3. **DocumentProcessor** (`src/rag/document_processor.py`)
- **Función**: Procesamiento y chunking de documentos
- **Capacidades**:
  - Extracción de texto de PDFs (PyPDF2)
  - División inteligente en chunks con solapamiento
  - Metadatos de categorización
  - Cache de documentos procesados

#### 4. **RAGManager** (`src/rag/rag_manager.py`)
- **Función**: Gestión de documentos y configuración
- **Responsabilidades**:
  - Control de versiones de documentos
  - Detección de cambios automática
  - Priorización de categorías
  - Validación de configuración

#### 5. **MemoryManager** (`src/rag/memory.py`)
- **Función**: Gestión de memoria conversacional
- **Características**:
  - Historial de consultas y respuestas
  - Contexto conversacional
  - Persistencia en disco
  - Limpieza automática de memoria antigua

#### 6. **PersonalityManager** (`src/rag/personality_manager.py`)
- **Función**: Gestión de personalidad y respuestas básicas
- **Capacidades**:
  - Patrones de activación
  - Clasificación de consultas
  - Respuestas predefinidas
  - Gestión de emociones

## **Flujo de Procesamiento**

### **1. Inicialización del Sistema**
```python
# El sistema se inicializa automáticamente
rag_system = RAGSystem()
rag_system.initialize()

# Procesa todos los documentos y crea embeddings
# Construye el vector store con FAISS
# Carga la memoria conversacional
```

### **2. Clasificación de Consultas**
El sistema clasifica automáticamente las consultas en tres categorías:

#### **Consultas Universitarias** (Alta Prioridad)
- **Palabras clave**: universidad, udit, carrera, titulación, grado, máster, doctorado, matrícula, inscripción, admisión, acceso, normativa, reglamento, política, calidad, medio ambiente, extinción, rac, condiciones, horarios, servicios, sede, campus, facultad, departamento, profesor, estudiante, alumno, examen, evaluación, nota, crédito, asignatura, materia, curso, semestre, académico, académica, docente, administrativo, secretaría, decano, rector, vicerrector, trabajo fin, tfg, tfm, tesis, proyecto, prácticas, internship, erasmus, intercambio, beca, ayuda, subvención, precio, coste, pago, factura, recibo, certificado, diploma, expediente, historial, calificaciones, notas, convocatoria, examen, evaluación, calificación

#### **Consultas Generales** (Baja Prioridad)
- **Palabras clave**: clima, tiempo, fecha, hora, distancia, luna, capital, país, ciudad, historia, ciencia, matemáticas, cálculo, suma, resta, multiplicación, división, juego, adivina, chiste, curiosidad, deporte, música, película, libro, arte, cocina, receta, viaje, turismo, hotel, restaurante, comida, bebida, salud, medicina, ejercicio, deporte, fútbol, baloncesto, tenis

#### **Consultas Desconocidas**
- Se procesan con confianza baja
- Se usan métodos de fallback

### **3. Procesamiento de Consultas Universitarias (RAG)**

#### **Búsqueda en Vector Store**
```python
# Búsqueda semántica por similitud
search_results = vector_store.search(query)

# Búsqueda por categoría específica
schedule_results = vector_store.search_by_category(query, "university_schedules")
```

#### **Generación de Respuestas Contextuales**
```python
# Construcción de contexto
context_text = "\n".join([
    f"Documento {i+1} ({result['category_name']}): {result['text']}"
    for i, result in enumerate(search_results[:3])
])

# Generación de respuesta con contexto
response = self._generate_rag_response(query, context_text, memory_context)
```

#### **Priorización de Horarios**
Para consultas sobre horarios, el sistema:
1. Detecta palabras clave de tiempo
2. Busca específicamente en la categoría `university_schedules`
3. Prioriza resultados de horarios en la respuesta
4. Proporciona interpretación contextual

### **4. Procesamiento de Consultas Generales (GPT)**
```python
# Fallback a GPT para preguntas no universitarias
response = self._generate_gpt_response(query, memory_context)
```

### **5. Gestión de Memoria**
```python
# Actualización de memoria conversacional
memory_manager.add_memory(query, response['answer'], query_embedding)

# Obtención de contexto previo
memory_context = memory_manager.get_context(query, query_embedding)
```

## **Configuración del Sistema**

### **Archivo de Configuración Principal** (`config/rag_config.json`)

#### **Documentos Configurados**
```json
{
    "documents": {
        "university_policies": {
            "name": "Políticas Universitarias",
            "priority": 1,
            "files": ["UDIT-POLITICA-DE-CALIDAD-Y-M.AMBIENTE.pdf"]
        },
        "academic_regulations": {
            "name": "Regulaciones Académicas", 
            "priority": 2,
            "files": ["04_Normativa_AccesoyAdmision.pdf", "NormativaExtincionTitulacionesOficiales.pdf", "06_Normativa-RAC.pdf"]
        },
        "general_conditions": {
            "name": "Condiciones Generales",
            "priority": 3,
            "files": ["Condiciones_generales24.25.pdf"]
        },
        "university_schedules": {
            "name": "Horarios y Servicios",
            "priority": 4,
            "files": ["Horarios_UDIT.txt"]
        }
    }
}
```

#### **Configuración RAG**
```json
{
    "rag_settings": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "max_context_chunks": 5,
        "similarity_threshold": 0.3,
        "max_results": 10,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_store": "faiss",
        "cache_dir": "./data/rag_cache"
    }
}
```

#### **Configuración de Fallback**
```json
{
    "fallback_settings": {
        "use_gpt_for_general": true,
        "use_gpt_for_unclear": true,
        "max_retries": 2,
        "confidence_threshold": 0.6
    }
}
```

## **Características Técnicas**

### **Modelo de Embeddings**
- **Modelo**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensión**: 384 vectores
- **Idioma**: Multilingüe (incluye español)
- **Performance**: Optimizado para velocidad y precisión

### **Vector Store (FAISS)**
- **Algoritmo**: Índice de similitud coseno
- **Búsqueda**: K-NN (K-Nearest Neighbors)
- **Cache**: Persistente en disco con timestamps
- **Actualización**: Incremental y automática

### **Chunking Inteligente**
- **Tamaño**: 1000 caracteres por chunk
- **Solapamiento**: 200 caracteres entre chunks
- **Preservación**: Límites de oraciones cuando es posible
- **Metadatos**: Categoría, archivo fuente, posición

### **Gestión de Memoria**
- **Tamaño máximo**: 100 elementos en memoria
- **Persistencia**: Archivos JSON con timestamps
- **Limpieza**: Automática cada 10 archivos
- **Contexto**: Últimas 2 consultas para continuidad

## **Casos de Uso Específicos**

### **1. Consultas sobre Horarios**
```
Usuario: "¿Cuáles son los horarios de la universidad?"
Sistema: 
1. Clasifica como "university" con alta confianza
2. Detecta palabras clave de tiempo
3. Busca en categoría "university_schedules"
4. Prioriza información de horarios
5. Genera respuesta contextual con interpretación
```

### **2. Consultas sobre Normativas**
```
Usuario: "¿Cuál es la normativa de admisión?"
Sistema:
1. Clasifica como "university" 
2. Busca en documentos académicos
3. Encuentra "04_Normativa_AccesoyAdmision.pdf"
4. Extrae chunks relevantes
5. Genera respuesta con contexto normativo
```

### **3. Consultas Generales**
```
Usuario: "¿Qué tiempo hace hoy?"
Sistema:
1. Clasifica como "general"
2. Usa fallback GPT
3. Proporciona respuesta genérica
4. Sugiere consultar documentos universitarios para temas específicos
```

## **Métricas y Monitoreo**

### **Estadísticas del Sistema**
```python
stats = rag_system.get_stats()
# Retorna:
{
    "system_initialized": true,
    "vector_store": {
        "total_chunks": 150,
        "categories": ["university_policies", "academic_regulations", "general_conditions", "university_schedules"],
        "embedding_dimension": 384
    },
    "rag_manager": {
        "total_documents": 6,
        "last_update": "2024-01-15T10:30:00",
        "cache_status": "valid"
    },
    "last_update_check": 1705312200.0
}
```

### **Logging y Debugging**
- **Nivel**: INFO por defecto
- **Formato**: Timestamp + Componente + Nivel + Mensaje
- **Archivos**: Rotación automática en `logs/`
- **Trazabilidad**: Consulta → Clasificación → Búsqueda → Respuesta

## **Optimizaciones y Mejoras**

### **1. Threshold Adaptativo**
- Ajuste automático del umbral de similitud
- Historial de probabilidades para optimización
- Adaptación a diferentes tipos de consultas

### **2. Cache Inteligente**
- Cache de embeddings por documento
- Cache de resultados de búsqueda frecuentes
- Invalidación automática por cambios en documentos

### **3. Priorización de Categorías**
- Sistema de prioridades numérico (1-999)
- Consultas urgentes (horarios, servicios)
- Documentos de referencia (políticas, normativas)

## **Dependencias y Requisitos**

### **Librerías Principales**
```python
# Procesamiento de documentos
PyPDF2>=3.0.0

# Embeddings y vectorización
sentence-transformers>=2.2.0
numpy>=1.21.0

# Vector store
faiss-cpu>=1.7.0

# Utilidades
pathlib>=1.0.1
logging>=0.5.1.2
```

### **Requisitos del Sistema**
- **Python**: 3.8+
- **Memoria**: Mínimo 4GB RAM
- **Almacenamiento**: 2GB para cache y modelos
- **CPU**: Compatible con FAISS

## **Mantenimiento y Actualización**

### **Actualización de Documentos**
```python
# Detección automática de cambios
if rag_manager.check_for_updates():
    rag_system.initialize(force_rebuild=True)

# Actualización manual
rag_system.update_documents()
```

### **Limpieza de Cache**
```python
# Limpieza de cache de documentos
document_processor.clear_cache()

# Limpieza de vector store
vector_store.clear()

# Limpieza de memoria
memory_manager.clear_old_memory()
```

### **Backup y Recuperación**
- **Vector Store**: Archivos `.pkl` con timestamps
- **Cache**: Directorio `data/rag_cache/`
- **Memoria**: Archivos JSON en `data/memory/`
- **Configuración**: Archivos JSON en `config/`

## **Troubleshooting Común**

### **1. Error de Modelo de Embeddings**
```python
# Solución: Instalación automática
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    subprocess.check_call(["pip", "install", "sentence-transformers"])
```

### **2. Documentos No Encontrados**
```python
# Verificar rutas en configuración
# Asegurar que archivos existen
# Verificar permisos de lectura
```

### **3. Memoria Insuficiente**
```python
# Reducir chunk_size
# Limpiar cache antiguo
# Optimizar modelo de embeddings
```

## **Conclusiones**

El Sistema RAG de UDI representa una implementación robusta y eficiente de recuperación aumentada de información, específicamente diseñada para el contexto universitario. Su arquitectura modular, sistema de clasificación inteligente y gestión de memoria conversacional lo convierten en una herramienta poderosa para proporcionar respuestas precisas y contextualmente relevantes sobre la documentación de UDIT.

El sistema demuestra excelente capacidad de adaptación, priorización inteligente de información y gestión eficiente de recursos, manteniendo la precisión y relevancia en las respuestas generadas.
