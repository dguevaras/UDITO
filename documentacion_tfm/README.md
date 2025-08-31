# UDI - Asistente de Reconocimiento Inteligente

## 🎯 Descripción

UDI es un asistente de voz inteligente diseñado para la Universidad UDIT. Funciona como el "cerebro" de un autómata, proporcionando información universitaria a través de conversaciones de voz naturales.

## 🚀 Características Principales

### 🎤 Sistema de Voz UDI
- **Wake Word**: "UDITO" para activar el asistente
- **Beep de Activación**: Sonido distintivo cuando se activa
- **Beep de Desactivación**: Sonido diferente cuando se desactiva
- **Escucha Continua**: Detecta voz y transcribe en tiempo real
- **Respuesta Rápida**: Procesa y responde inmediatamente

### 🧠 Inteligencia Artificial
- **RAG (Retrieval-Augmented Generation)**: Acceso a documentación universitaria
- **Personalidad Humanizada**: Respuestas naturales y expresivas
- **Clasificación Inteligente**: Distingue entre tipos de preguntas
- **Memoria de Conversación**: Mantiene contexto de interacciones

### 🔊 Sistema de Audio
- **STT (Speech-to-Text)**: Faster-Whisper para transcripción precisa
- **TTS (Text-to-Speech)**: Piper TTS con voz natural en español
- **Detección de Voz**: WebRTC VAD para detección eficiente

### 🎯 **Wake Word Integrado con Wakeword-detector**

## 📁 Estructura del Proyecto

```
udi_rag/
├── config/                     # Configuraciones
│   ├── rag_config.json        # Configuración RAG
│   ├── settings.json          # Configuración general
│   ├── tts_config.json        # Configuración TTS
│   └── wake_word_config.json  # Configuración Wake Word
├── data/                      # Datos y caché
│   ├── qa/                   # Preguntas y respuestas
│   ├── rag_cache/            # Caché del sistema RAG
│   └── memory/               # Memoria de conversaciones
├── docs/                     # Documentación universitaria
├── piper/                    # Sistema TTS Piper
├── WakeWordProject/          # Proyecto de entrenamiento wakeword-detector
│   ├── models/               # Modelos entrenados
│   ├── data/                 # Datos de entrenamiento
│   │   ├── wake-word/        # 40 muestras de "UDITO"
│   │   └── not-wake-word/    # 125 muestras negativas
│   └── scripts/              # Scripts de entrenamiento
├── src/                      # Código fuente
│   ├── rag/                  # Sistema RAG
│   │   ├── core.py           # Funcionalidad principal
│   │   ├── rag_system.py     # Sistema RAG completo
│   │   ├── personality_manager.py  # Gestor de personalidad
│   │   └── vector_store.py   # Almacén de vectores
│   ├── tts/                  # Text-to-Speech
│   │   ├── piper_tts_real.py # Implementación Piper
│   │   └── azure_tts.py      # Implementación Azure (opcional)
│   ├── voice/                # Sistema de voz
│   │   ├── detector_whisper_faster.py  # Detector STT
│   │   ├── audio_handler_faster.py     # Manejo de audio
│   │   └── audio_recorder.py           # Grabación de audio
│   ├── wake_word/            # Módulo Wake Word
│   │   ├── __init__.py       # Inicialización del módulo
│   │   ├── hybrid_wake_word_detector.py  # Detector híbrido
│   │   └── detector_hmm_gmm_professional.py # Detector HMM/GMM (funcional)
│   └── utils/                # Utilidades
│       └── logger.py         # Sistema de logging
├── main.py                   # 🎯 SISTEMA PRINCIPAL INTEGRADO
├── test_hybrid_detector.py   # 🆕 Script de pruebas del detector
└── requirements.txt          # Dependencias actualizadas
```

## 🎯 Sistema Principal: `main.py` (INTEGRADO)

Este es el sistema principal que integra el wake word entrenado con wakeword-detector:

### Funcionamiento
1. **Escucha Pasiva**: Espera a que digas "UDITO"
2. **Activación**: Detecta el wake word usando similitud MFCC
3. **Escucha Activa**: Escucha tu pregunta
4. **Procesamiento**: Clasifica y busca respuesta
5. **Respuesta**: Responde con voz
6. **Desactivación**: Se desactiva automáticamente

### Modos de Operación
```bash
# Modo de wake word (recomendado)
python main.py --mode wake_word

# Modo de prueba del sistema
python main.py --mode test

# Mostrar ayuda
python main.py --help
```

### 🧪 **Pruebas del Detector Híbrido**
```bash
# Probar el detector de wake word
python test_hybrid_detector.py
```

## 🔧 **Wake Word: Integración wakeword-detector**

### **Características del Sistema:**
- **Muestras de Entrenamiento**: 150 grabaciones de "UDITO"
- **Muestras Negativas**: 160 grabaciones de palabras que NO son "UDITO"
- **Tecnología**: Detector híbrido HMM/GMM entrenado con wakeword-detector
- **Método**: Similitud MFCC con muestras de referencia
- **Precisión**: Ajustable mediante threshold de confianza

### **Archivos de Configuración:**
- **WakeWordProject/config/wake_word_config.json**: Configuración del wake word
- **WakeWordProject/data/tags.txt**: Etiquetas de entrenamiento
- **WakeWordProject/data/wake-word/**: Muestras de "UDITO"
- **WakeWordProject/data/not-wake-word/**: Muestras negativas

### **Entrenamiento:**
```bash
cd WakeWordProject
python WakeWordProject/train_hmm_gmm_professional.py
```

## 🚀 **Instalación y Uso**

### **Requisitos:**
- Python 3.8+
- wakeword-detector (entorno virtual dedicado)
- PyAudio
- Librosa
- TensorFlow 2.x

### **Instalación:**
```bash
# Clonar repositorio
git clone <repository-url>
cd udi_rag

# Instalar dependencias
pip install -r requirements.txt

# Activar entorno wakeword-detector
cd WakeWordProject
env_wakeword_detector\Scripts\activate
```

### **Uso Rápido:**
```bash
# Ejecutar sistema completo
python main.py --mode wake_word

# Probar detector
python main.py --mode test

# Verificar integración
python test_hybrid_detector.py
```

## 📊 **Estado del Proyecto**

### ✅ **COMPLETADO:**
- Sistema wake word integrado con wakeword-detector
- Detector híbrido compatible con versiones modernas
- 40 muestras de "UDITO" cargadas y funcionales
- Sistema principal completamente integrado
- Pruebas de integración exitosas

### 🔄 **EN DESARROLLO:**
- Optimización para Jetson Nano
- Pruebas de campo en entornos ruidosos

## 🤝 **Contribución**

Este proyecto está diseñado para la Universidad UDIT. Para contribuir:

1. **Fork** el repositorio
2. **Crea** una rama para tu feature
3. **Commit** tus cambios
4. **Push** a la rama
5. **Abre** un Pull Request

## 📄 **Licencia**

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

---

**UDI - Unidad Digital Interactiva**  
*Asistente de voz inteligente para la Universidad UDIT*
