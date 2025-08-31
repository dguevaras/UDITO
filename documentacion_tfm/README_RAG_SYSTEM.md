# 🧠 Sistema RAG de UDI - README

## **Descripción**

El **Sistema RAG (Retrieval-Augmented Generation)** de UDI es el núcleo inteligente que permite al asistente de voz responder preguntas sobre documentación universitaria de UDIT con precisión y contexto. Combina **recuperación semántica avanzada** con **generación de respuestas inteligentes**.

## **🚀 Características Principales**

- **🔍 Búsqueda Semántica**: Usa embeddings vectoriales para encontrar información relevante
- **📚 Procesamiento de PDFs**: Extrae y procesa automáticamente documentos universitarios
- **🎯 Clasificación Inteligente**: Distingue entre consultas universitarias y generales
- **💾 Memoria Conversacional**: Mantiene contexto de conversaciones previas
- **⚡ Cache Inteligente**: Optimiza rendimiento con almacenamiento persistente
- **🔄 Actualización Automática**: Detecta cambios en documentos y se actualiza

## **🏗️ Arquitectura**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RAGSystem     │    │  VectorStore    │    │ DocumentProcessor│
│   (Orquestador) │◄──►│   (FAISS)       │◄──►│   (PDF + TXT)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  RAGManager     │    │ MemoryManager    │    │PersonalityManager│
│ (Documentos)    │    │ (Conversación)   │    │ (Personalidad)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## **📦 Instalación**

### **Requisitos Previos**
```bash
# Python 3.8+
python --version

# Memoria: Mínimo 4GB RAM
# Almacenamiento: 2GB para cache y modelos
```

### **Instalación de Dependencias**
```bash
# Instalar librerías principales
pip install sentence-transformers>=2.2.0
pip install faiss-cpu>=1.7.4
pip install PyPDF2>=3.0.0
pip install numpy>=1.21.0

# O instalar desde requirements.txt
pip install -r requirements.txt
```

## **⚙️ Configuración**

### **1. Archivo de Configuración Principal**
```json
// config/rag_config.json
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
            "files": ["04_Normativa_AccesoyAdmision.pdf"]
        }
    },
    "rag_settings": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_store": "faiss",
        "cache_dir": "./data/rag_cache"
    }
}
```

### **2. Estructura de Directorios**
```
ari_rag/
├── src/rag/                    # Código fuente del sistema RAG
├── config/                     # Archivos de configuración
├── docs/                       # Documentos PDF de la universidad
├── data/                       # Cache y datos del sistema
│   ├── rag_cache/             # Cache de embeddings
│   └── memory/                # Memoria conversacional
└── logs/                       # Logs del sistema
```

## **🚀 Uso Rápido**

### **1. Inicialización Básica**
```python
from src.rag.rag_system import RAGSystem

# Crear instancia del sistema
rag_system = RAGSystem()

# Inicializar (procesa documentos y crea embeddings)
rag_system.initialize()

# Verificar estado
stats = rag_system.get_stats()
print(f"Sistema listo: {stats['vector_store']['total_chunks']} chunks")
```

### **2. Procesar Consultas**
```python
# Consulta universitaria (usa RAG)
response = rag_system.process_query("¿Cuáles son los horarios de la universidad?")
print(f"Respuesta: {response['answer']}")
print(f"Tipo: {response['classification']['query_type']}")

# Consulta general (usa fallback)
response = rag_system.process_query("¿Qué tiempo hace hoy?")
print(f"Respuesta: {response['answer']}")
```

### **3. Búsqueda Directa en Vector Store**
```python
from src.rag.vector_store import VectorStore

vector_store = VectorStore()
vector_store.load()

# Búsqueda semántica
results = vector_store.search("normativa de admisión")
for result in results[:3]:
    print(f"Documento: {result['source_file']}")
    print(f"Texto: {result['text'][:100]}...")
```

## **🔧 Funcionalidades Avanzadas**

### **Clasificación Inteligente de Consultas**

El sistema clasifica automáticamente las consultas:

#### **Consultas Universitarias** (Alta Prioridad)
- **Palabras clave**: universidad, udit, carrera, titulación, grado, máster, doctorado, matrícula, inscripción, admisión, acceso, normativa, reglamento, política, calidad, medio ambiente, extinción, rac, condiciones, horarios, servicios, sede, campus, facultad, departamento, profesor, estudiante, alumno, examen, evaluación, nota, crédito, asignatura, materia, curso, semestre, académico, académica, docente, administrativo, secretaría, decano, rector, vicerrector, trabajo fin, tfg, tfm, tesis, proyecto, prácticas, internship, erasmus, intercambio, beca, ayuda, subvención, precio, coste, pago, factura, recibo, certificado, diploma, expediente, historial, calificaciones, notas, convocatoria, examen, evaluación, calificación

#### **Consultas Generales** (Baja Prioridad)
- **Palabras clave**: clima, tiempo, fecha, hora, distancia, luna, capital, país, ciudad, historia, ciencia, matemáticas, cálculo, suma, resta, multiplicación, división, juego, adivina, chiste, curiosidad, deporte, música, película, libro, arte, cocina, receta, viaje, turismo, hotel, restaurante, comida, bebida, salud, medicina, ejercicio, deporte, fútbol, baloncesto, tenis

### **Priorización de Categorías**
```python
# Sistema de prioridades (1 = más alta, 999 = más baja)
priorities = {
    "university_policies": 1,      # Políticas (más importante)
    "academic_regulations": 2,     # Regulaciones académicas
    "general_conditions": 3,       # Condiciones generales
    "university_schedules": 4      # Horarios y servicios
}
```

### **Gestión de Memoria Conversacional**
```python
from src.rag.memory import MemoryManager

memory = MemoryManager()

# Agregar consulta a memoria
memory.add_memory("¿Cuáles son los horarios?", "Los horarios son...", embedding)

# Obtener contexto previo
context = memory.get_context("nueva consulta", new_embedding)
```

## **📊 Monitoreo y Métricas**

### **Estadísticas del Sistema**
```python
stats = rag_system.get_stats()
print(json.dumps(stats, indent=2))

# Salida ejemplo:
{
    "system_initialized": true,
    "vector_store": {
        "total_chunks": 150,
        "categories": ["university_policies", "academic_regulations"],
        "embedding_dimension": 384
    },
    "rag_manager": {
        "total_documents": 6,
        "last_update": "2024-01-15T10:30:00",
        "cache_status": "valid"
    }
}
```

### **Logging y Debugging**
```python
import logging

# Configurar nivel de logging
logging.basicConfig(level=logging.INFO)

# Los logs se guardan en logs/ con rotación automática
# Formato: Timestamp + Componente + Nivel + Mensaje
```

## **🔄 Mantenimiento**

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
# Limpiar cache de documentos
document_processor.clear_cache()

# Limpiar vector store
vector_store.clear()

# Limpiar memoria antigua
memory_manager.clear_old_memory()
```

### **Backup y Recuperación**
- **Vector Store**: Archivos `.pkl` con timestamps
- **Cache**: Directorio `data/rag_cache/`
- **Memoria**: Archivos JSON en `data/memory/`
- **Configuración**: Archivos JSON en `config/`

## **🐛 Troubleshooting**

### **Problemas Comunes**

#### **1. Error de Modelo de Embeddings**
```bash
# Solución: Instalación manual
pip install sentence-transformers>=2.2.0

# Verificar instalación
python -c "from sentence_transformers import SentenceTransformer; print('OK')"
```

#### **2. Documentos No Encontrados**
```bash
# Verificar rutas en config/rag_config.json
# Asegurar que archivos existen en docs/
# Verificar permisos de lectura
ls -la docs/
```

#### **3. Memoria Insuficiente**
```bash
# Reducir chunk_size en configuración
# Limpiar cache antiguo
rm -rf data/rag_cache/*
# Reiniciar sistema
```

#### **4. Error de FAISS**
```bash
# Reinstalar FAISS
pip uninstall faiss-cpu
pip install faiss-cpu>=1.7.4

# Para Windows, usar versión compatible
pip install faiss-cpu==1.7.4
```

## **📈 Optimización**

### **Parámetros Ajustables**
```json
{
    "rag_settings": {
        "chunk_size": 1000,           // Tamaño de chunks (caracteres)
        "chunk_overlap": 200,          // Solapamiento entre chunks
        "similarity_threshold": 0.3,   // Umbral de similitud
        "max_results": 10,             // Máximo resultados de búsqueda
        "max_context_chunks": 5        // Máximo chunks en contexto
    }
}
```

### **Threshold Adaptativo**
```python
# El sistema ajusta automáticamente el umbral basado en:
# - Historial de probabilidades
# - Tipo de consulta
# - Calidad de resultados previos
```

## **🔗 Integración con Otros Sistemas**

### **Pipeline de Voz UDI**
```
Wake Word → Activación → STT → RAG → TTS → Respuesta
```

### **API REST (Futuro)**
```python
# Endpoint para consultas
POST /api/rag/query
{
    "query": "¿Cuáles son los horarios?",
    "user_id": "user123",
    "session_id": "session456"
}
```

## **📚 Documentación Adicional**

- **Documentación Completa**: `DOCUMENTACION_SISTEMA_RAG.md`
- **Código Fuente**: `src/rag/`
- **Configuración**: `config/rag_config.json`
- **Logs**: `logs/`

## **🤝 Contribución**

### **Estructura del Código**
- **Módulos independientes** para fácil mantenimiento
- **Interfaces claras** entre componentes
- **Logging detallado** para debugging
- **Configuración externa** para flexibilidad

### **Estándares de Código**
- **Python 3.8+** con type hints
- **Docstrings** en todas las funciones
- **Manejo de errores** robusto
- **Tests unitarios** (futuro)

## **📄 Licencia**

Este sistema RAG es parte del proyecto UDI y está sujeto a la licencia del proyecto principal.

## **📞 Soporte**

Para problemas técnicos o preguntas sobre el sistema RAG:
1. Revisar logs en `logs/`
2. Verificar configuración en `config/rag_config.json`
3. Consultar documentación completa
4. Revisar troubleshooting común

---

**🎯 El Sistema RAG de UDI: Inteligencia artificial para respuestas universitarias precisas y contextuales.**
