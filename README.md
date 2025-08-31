# 🎓 UDI - Asistente de Voz Inteligente para Universidad

## 📋 Descripción

**UDI** (Asistente de Reconocimiento Inteligente) es un sistema de asistente de voz completo diseñado específicamente para universidades. Integra detección de palabra de activación, procesamiento de lenguaje natural, sistema RAG (Retrieval-Augmented Generation) y síntesis de voz para proporcionar respuestas inteligentes sobre información universitaria.

## 🏗️ Arquitectura del Sistema

```
🎤 Wake Word Detection → 🗣️ STT (Faster-Whisper) → 🧠 RAG/NLP (Gemma 2B) → 🔊 TTS (Piper)
```

### Componentes Principales

- **🔊 Wake Word Detection**: Sistema personalizado para detectar "UDITO"
- **🎤 Speech-to-Text**: Faster-Whisper para transcripción de voz
- **🧠 RAG/NLP**: Sistema de recuperación y generación con Gemma 2B
- **🔊 Text-to-Speech**: Piper TTS para síntesis de voz natural
- **📚 Document Processing**: Procesamiento automático de documentos PDF
- **💾 Memory System**: Sistema de memoria conversacional

## 🚀 Características

### ✅ Funcionalidades Implementadas

- **Detección de Palabra de Activación**: "UDITO" con alta precisión
- **Procesamiento de Documentos**: PDFs universitarios automáticamente
- **Respuestas Inteligentes**: Basadas en documentos oficiales
- **Síntesis de Voz Natural**: Voz en español con Piper TTS
- **Sistema de Memoria**: Recuerda conversaciones previas
- **Respuestas Básicas**: Para preguntas frecuentes y saludos
- **Optimización de Rendimiento**: Gemma 2B optimizado para velocidad

### 📊 Capacidades del Sistema

- **91 chunks** de documentos procesados
- **4 categorías** de información universitaria
- **Respuestas en tiempo real** con latencia < 2 segundos
- **Soporte completo en español**
- **Compatibilidad Windows y Jetson Nano**

## 📁 Estructura del Proyecto

```
UDI-final/
├── 📁 src/
│   ├── 🎤 voice/           # STT con Faster-Whisper
│   ├── 🧠 rag/             # Sistema RAG con Gemma 2B
│   ├── 🔊 tts/             # Piper TTS
│   └── 🔧 utils/           # Utilidades
├── 📁 config/              # Configuraciones
├── 📁 data/
│   ├── 📚 qa/              # Respuestas básicas
│   └── 💾 rag_cache/       # Cache del sistema RAG
├── 📁 docs/                # Documentos universitarios
├── 📁 documentacion_tfm/   # Documentación del TFM
├── 🎯 main_*.py            # Archivos principales
└── 📋 README.md
```

## 🛠️ Instalación

### Prerrequisitos

- Python 3.11+
- Windows 10/11 o Jetson Nano
- 8GB RAM mínimo
- Espacio en disco: 5GB

### Instalación Rápida

```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/UDI-final.git
cd UDI-final

# 2. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar Hugging Face (para Gemma 2B)
huggingface-cli login
# O configurar token en src/rag/gemma_llm.py
```

### Configuración Adicional

1. **Hugging Face Token**: Configurar para Gemma 2B
   ```bash
   # Copiar archivo de ejemplo
   cp config/huggingface_token.example.txt config/huggingface_token.txt
   # Editar y agregar tu token real
   ```

2. **Piper TTS**: Descargar modelo de voz español
3. **Documentos**: Colocar PDFs universitarios en `docs/`
4. **Wake Word**: El modelo ya está incluido

## 🎮 Uso

### Archivos Principales

- **`main_console.py`**: Interfaz de consola para pruebas
- **`main_simple.py`**: Sistema completo con wake word
- **`main_wakeword.py`**: Probador de detección de palabra

### Ejemplo de Uso

```bash
# Sistema completo con wake word
python main_simple.py

# Interfaz de consola para pruebas
python main_console.py

# Probar detección de palabra
python main_wakeword.py
```

### Comandos de Voz

- **"UDITO"** → Activa el sistema
- **"¿Cuáles son los horarios?"** → Información de horarios
- **"¿Qué es UDIT?"** → Información de la universidad
- **"¿Cómo te llamas?"** → Información del asistente

## 🔧 Configuración

### Archivos de Configuración

- `config/rag_config.json`: Configuración del sistema RAG
- `config/tts_config.json`: Configuración de Piper TTS
- `config/settings.json`: Configuración general
- `data/qa/basic_qa.json`: Respuestas básicas

### Personalización

1. **Wake Word**: Modificar en `WakeWord-project/`
2. **Voz TTS**: Cambiar modelo en `config/tts_config.json`
3. **Documentos**: Agregar PDFs en `docs/`
4. **Respuestas**: Editar `data/qa/basic_qa.json`

## 📊 Rendimiento

### Métricas del Sistema

- **Detección Wake Word**: < 100ms
- **Respuesta RAG**: < 2 segundos
- **Síntesis de Voz**: < 1 segundo
- **Precisión STT**: > 95%
- **Memoria**: 91 chunks procesados

### Optimizaciones Implementadas

- **Gemma 2B**: Quantización 8-bit para velocidad
- **Cache de Respuestas**: Respuestas frecuentes en memoria
- **Procesamiento Paralelo**: STT y TTS no bloqueantes
- **Compresión de Modelos**: Modelos optimizados para CPU

## 🔍 Troubleshooting

### Problemas Comunes

1. **Error de Hugging Face**: Configurar token de autenticación
2. **Audio no funciona**: Verificar drivers de audio
3. **Wake word no detecta**: Ajustar sensibilidad en configuración
4. **RAG lento**: Verificar conexión a internet para Gemma 2B

### Logs y Debug

- Los logs se muestran en consola con timestamps
- Usar `main_wakeword.py` para debug de detección
- Verificar archivos de cache en `data/rag_cache/`

## 📚 Documentación Técnica

### Documentación Completa

- `documentacion_tfm/DOCUMENTACION_SISTEMA_WAKEWORD.md`: Detalles técnicos del wake word
- `documentacion_tfm/DOCUMENTACION_RAG_GEMMA.md`: Implementación de RAG con Gemma 2B
- `ARQUITECTURA_PIPELINE_UDI.md`: Arquitectura completa del sistema

### Tecnologías Utilizadas

- **Wake Word**: TensorFlow Lite, OpenWakeWord
- **STT**: Faster-Whisper
- **RAG**: Sentence Transformers, FAISS
- **LLM**: Google Gemma 2B Instruction Tuned
- **TTS**: Piper TTS
- **Audio**: PyAudio, SoundDevice

## 🤝 Contribución

Este proyecto es parte de un TFM (Trabajo de Fin de Máster). Para contribuciones:

1. Fork el repositorio
2. Crear rama para feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 👨‍🎓 Autor

**UDI - Asistente de Reconocimiento Inteligente**
- **Proyecto**: TFM - Máster en Inteligencia Artificial
- **Universidad**: UDIT (Universidad de Tecnología e Innovación)
- **Año**: 2025

## 🙏 Agradecimientos

- **Google**: Gemma 2B Instruction Tuned
- **Piper TTS**: Síntesis de voz
- **Faster-Whisper**: Transcripción de voz
- **OpenWakeWord**: Detección de palabra de activación
- **Hugging Face**: Transformers y modelos

---

**🎯 UDI - Tu Asistente Universitario Inteligente**
