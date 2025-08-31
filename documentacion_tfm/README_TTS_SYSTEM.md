# 🔊 Sistema TTS de UDI - README

## **Descripción**

El **Sistema TTS (Text-to-Speech)** de UDI utiliza **Piper TTS** para convertir respuestas del asistente en voz natural y clara. Piper es un motor de síntesis de voz offline de alta calidad que proporciona voces naturales en español, perfecto para un asistente universitario.

## **🚀 Características Principales**

- **🎤 Voces Naturales**: Modelos ONNX de alta calidad en español
- **🌍 Multilingüe**: Soporte para español de España y México
- **💻 Funcionamiento Offline**: No requiere conexión a internet
- **⚡ Alta Calidad**: Síntesis de voz clara y natural
- **🔧 Configurable**: Ajustes de velocidad, volumen y personalidad
- **🎯 Integrado**: Perfectamente integrado con el pipeline UDI

## **🏗️ Arquitectura del Sistema**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PiperTTS      │    │   Piper.exe     │    │   Modelos ONNX  │
│   (Controlador) │◄──►│   (Motor TTS)   │◄──►│   (Voces)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Configuración  │    │  eSpeak-NG      │    │  Reproducción   │
│   (JSON)        │    │ (Fonetización)  │    │  (Pygame)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## **📦 Instalación**

### **Requisitos Previos**
```bash
# Python 3.8+
python --version

# Dependencias de audio
pip install pygame>=2.0.0
pip install playsound>=1.2.2  # Fallback opcional
```

### **Componentes de Piper**
El sistema ya incluye todos los componentes necesarios en la carpeta `piper/`:

```
piper/
├── piper.exe                    # Motor TTS principal
├── piper_phonemize.dll         # DLL de fonetización
├── espeak-ng.dll               # DLL de eSpeak-NG
├── onnxruntime.dll             # Runtime de ONNX
├── voices/                      # Modelos de voz
│   ├── es_ES-carlfm-x_low.onnx
│   ├── es_ES-sharvard-medium.onnx
│   ├── es_MX-ald-medium.onnx
│   └── *.onnx.json             # Configuraciones de voz
└── espeak-ng-data/             # Datos de fonetización
```

## **⚙️ Configuración**

### **Archivo de Configuración Principal**
```json
// config/tts_config.json
{
  "voice_rate": 150,
  "voice_volume": 0.9,
  "voice_id": "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_ES-MX_SABINA_11.0",
  "assistant_name": "UDI",
  "greeting": "Hola, soy UDI, tu asistente universitario. ¿En qué puedo ayudarte?",
  "farewell": "Gracias por usar UDI. ¡Que tengas un buen día!",
  "thinking": "Déjame buscar esa información...",
  "not_found": "Lo siento, no encontré información sobre eso.",
  "error": "Tuve un problema al procesar tu consulta."
}
```

### **Parámetros Configurables**
- **`voice_rate`**: Velocidad de habla (100-200)
- **`voice_volume`**: Volumen de salida (0.0-1.0)
- **`assistant_name`**: Nombre del asistente
- **`greeting`**: Mensaje de saludo personalizado
- **`farewell`**: Mensaje de despedida
- **`thinking`**: Mensaje mientras procesa
- **`not_found`**: Respuesta cuando no encuentra información
- **`error`**: Mensaje de error

## **🚀 Uso Rápido**

### **1. Inicialización Básica**
```python
from src.tts.piper_tts_real import PiperTTS

# Crear instancia del TTS
tts = PiperTTS()

# Verificar voces disponibles
tts.list_available_voices()

# Probar voz del sistema
tts.test_voice()
```

### **2. Síntesis de Texto Personalizado**
```python
# Reproducir texto personalizado
tts.speak("Hola, ¿cómo estás hoy?")

# Reproducir con espera
tts.speak("Este mensaje se reproduce completamente", wait=True)

# Reproducir sin espera (asíncrono)
tts.speak("Este mensaje se reproduce en segundo plano", wait=False)
```

### **3. Respuestas Pre-entrenadas**
```python
# Usar respuestas del sistema
tts.speak_response("greeting")      # Saludo inicial
tts.speak_response("thinking")      # Mientras procesa
tts.speak_response("not_found")     # No encontró información
tts.speak_response("farewell")      # Despedida
tts.speak_response("error")         # Mensaje de error
```

### **4. Cambio de Voz**
```python
# Cambiar a voz específica
tts.test_voice("es_ES-sharvard-medium.onnx")

# Listar todas las voces disponibles
tts.list_available_voices()
```

## **🎤 Modelos de Voz Disponibles**

### **Voces en Español (España)**
- **`es_ES-carlfm-x_low.onnx`** (60MB) - Voz masculina, calidad baja
- **`es_ES-mls_9972-low.onnx`** (60MB) - Voz femenina, calidad baja
- **`es_ES-sharvard-medium.onnx`** (73MB) - Voz masculina, calidad media

### **Voces en Español (México)**
- **`es_MX-ald-medium.onnx`** (60MB) - Voz femenina, calidad media

### **Características de Calidad**
- **`low`**: Calidad básica, archivo más pequeño
- **`medium`**: Calidad mejorada, archivo intermedio
- **`x_low`**: Calidad mínima, archivo más pequeño

## **🔧 Funcionalidades Avanzadas**

### **Ajuste Automático de Pronunciación**
```python
def _adjust_text_for_pronunciation(self, text: str) -> str:
    # Normaliza "UDI" para pronunciación natural
    name_pron = self.config.get('assistant_name', 'Udi')
    if name_pron.strip().lower() == 'udi':
        text = text.replace('UDI', 'Udi').replace('U.D.I.', 'Udi')
    elif name_pron.strip().lower() == 'udito':
        text = text.replace('UDITO', 'Udito')
    return text
```

### **Gestión de Respuestas Dinámicas**
```python
def _get_responses(self) -> Dict[str, list]:
    assistant_name = self.config.get('assistant_name', 'UDI')
    
    return {
        "greeting": [
            f"Hola, soy {assistant_name} de la universidad UDIT...",
            f"¡Hola! Soy {assistant_name}, tu asistente universitario..."
        ],
        "thinking": [
            "Déjame buscar esa información...",
            "Un momento, estoy consultando los documentos...",
            "Permíteme buscar en la base de datos..."
        ]
        # ... más respuestas
    }
```

### **Reproducción de Audio Robusta**
```python
def _play_audio(self, audio_path: str):
    try:
        # Intenta usar pygame (recomendado)
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        
        # Espera a que termine
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
    except ImportError:
        # Fallback a playsound
        try:
            from playsound import playsound
            playsound(audio_path)
        except ImportError:
            logger.warning("No se pudo reproducir audio")
```

## **📊 Monitoreo y Debugging**

### **Logging del Sistema**
```python
import logging

# Configurar nivel de logging
logging.basicConfig(level=logging.INFO)

# Los logs muestran:
# - Inicialización de voces
# - Procesamiento de texto
# - Errores de síntesis
# - Estado de reproducción
```

### **Verificación de Estado**
```python
# Verificar que Piper esté disponible
if not Path("piper/piper.exe").exists():
    print("❌ Piper TTS no encontrado")
else:
    print("✅ Piper TTS disponible")

# Verificar voces disponibles
voices = list(Path("piper/voices").glob("*.onnx"))
print(f"🎤 Voces disponibles: {len(voices)}")
```

## **🔄 Mantenimiento**

### **Actualización de Voces**
```bash
# Descargar nuevas voces desde:
# https://huggingface.co/rhasspy/piper-voices

# Colocar en piper/voices/
# Asegurar que tengan archivo .onnx.json correspondiente
```

### **Limpieza de Archivos Temporales**
```python
# El sistema limpia automáticamente archivos temporales
# Pero puedes limpiar manualmente si es necesario:
import tempfile
import os

# Limpiar archivos temporales del sistema
tempfile.tempdir = None  # Usar directorio por defecto
```

### **Verificación de Dependencias**
```bash
# Verificar DLLs de Piper
ls -la piper/*.dll

# Verificar modelo ONNX
python -c "import onnx; print('ONNX disponible')"

# Verificar pygame
python -c "import pygame; print('Pygame disponible')"
```

## **🐛 Troubleshooting**

### **Problemas Comunes**

#### **1. Piper no encontrado**
```bash
# Verificar que piper.exe existe
ls -la piper/piper.exe

# Verificar permisos de ejecución
chmod +x piper/piper.exe

# En Windows, verificar que no esté bloqueado
# Click derecho → Propiedades → Desbloquear
```

#### **2. Error de voz no encontrada**
```bash
# Verificar archivos de voz
ls -la piper/voices/*.onnx

# Verificar archivos de configuración
ls -la piper/voices/*.onnx.json

# Reinstalar voz si es necesario
# Descargar desde HuggingFace
```

#### **3. Error de reproducción de audio**
```bash
# Instalar pygame
pip install pygame>=2.0.0

# Instalar playsound como fallback
pip install playsound>=1.2.2

# Verificar drivers de audio del sistema
# Verificar volumen del sistema
```

#### **4. Error de memoria insuficiente**
```bash
# Usar voces de menor calidad
# Cambiar a es_ES-carlfm-x_low.onnx
# Reducir chunk_size en configuración
```

### **Diagnóstico del Sistema**
```python
def diagnose_tts_system():
    """Diagnóstico completo del sistema TTS"""
    print("🔍 DIAGNÓSTICO DEL SISTEMA TTS")
    print("=" * 40)
    
    # Verificar Piper
    piper_path = Path("piper/piper.exe")
    print(f"Piper.exe: {'✅' if piper_path.exists() else '❌'}")
    
    # Verificar voces
    voices_path = Path("piper/voices")
    voices = list(voices_path.glob("*.onnx"))
    print(f"Modelos de voz: {len(voices)}")
    
    # Verificar dependencias
    try:
        import pygame
        print("Pygame: ✅")
    except ImportError:
        print("Pygame: ❌")
    
    try:
        from playsound import playsound
        print("Playsound: ✅")
    except ImportError:
        print("Playsound: ❌")
```

## **📈 Optimización**

### **Parámetros de Rendimiento**
```json
{
  "voice_rate": 150,           // Velocidad óptima para claridad
  "voice_volume": 0.9,         // Volumen alto para audibilidad
  "chunk_size": 1000,          // Tamaño de texto por chunk
  "max_duration": 30           // Duración máxima por síntesis
}
```

### **Selección de Voz**
- **Para desarrollo**: `es_ES-carlfm-x_low.onnx` (rápida)
- **Para producción**: `es_ES-sharvard-medium.onnx` (calidad)
- **Para español mexicano**: `es_MX-ald-medium.onnx`

### **Gestión de Memoria**
```python
# Limpiar archivos temporales inmediatamente
# Usar voces de menor tamaño para sistemas con poca RAM
# Implementar cola de síntesis para textos largos
```

## **🔗 Integración con Otros Sistemas**

### **Pipeline de Voz UDI**
```
Wake Word → Activación → STT → RAG → TTS (Piper) → Respuesta
```

### **Integración con RAG**
```python
# El sistema TTS recibe respuestas del RAG
response = rag_system.process_query("¿Cuáles son los horarios?")

# Sintetiza la respuesta
tts.speak(response['answer'])

# O usa respuestas pre-entrenadas
if response['source'] == 'rag_no_results':
    tts.speak_response("not_found")
```

### **API REST (Futuro)**
```python
# Endpoint para síntesis de voz
POST /api/tts/speak
{
    "text": "Texto a sintetizar",
    "voice": "es_ES-sharvard-medium.onnx",
    "rate": 150,
    "volume": 0.9
}
```

## **📚 Documentación Adicional**

- **Código Fuente**: `src/tts/piper_tts_real.py`
- **Configuración**: `config/tts_config.json`
- **Ejecutables**: `piper/piper.exe`
- **Modelos de Voz**: `piper/voices/`
- **Documentación Piper**: https://github.com/rhasspy/piper

## **🤝 Contribución**

### **Estructura del Código**
- **Clase principal** `PiperTTS` bien documentada
- **Manejo de errores** robusto
- **Configuración externa** para flexibilidad
- **Fallbacks** para diferentes entornos

### **Estándares de Código**
- **Python 3.8+** con type hints
- **Docstrings** en todas las funciones
- **Logging detallado** para debugging
- **Manejo de excepciones** completo

## **📄 Licencia**

Este sistema TTS es parte del proyecto UDI y está sujeto a la licencia del proyecto principal. Piper TTS tiene su propia licencia MIT.

## **📞 Soporte**

Para problemas técnicos o preguntas sobre el sistema TTS:
1. Revisar logs del sistema
2. Verificar configuración en `config/tts_config.json`
3. Verificar archivos en `piper/`
4. Consultar documentación de Piper
5. Revisar troubleshooting común

---

**🎯 El Sistema TTS de UDI: Voz natural y clara para un asistente universitario profesional.**
