# 🔗 INTEGRACIÓN DEL WAKE WORD AL PIPELINE UDI - ANÁLISIS TÉCNICO CORREGIDO

## **RESUMEN EJECUTIVO**

Este documento analiza la **integración actual** del modelo de wake word entrenado (HMM/GMM) al pipeline principal de UDI y propone la **arquitectura de integración correcta** que debería implementarse para lograr un sistema unificado y eficiente.

**⚠️ CORRECCIÓN IMPORTANTE**: El sistema de entrenamiento se cambió de Mycroft Precise a **wakeword-detector**, por lo que todas las referencias anteriores a Mycroft son incorrectas.

## **1. ESTADO ACTUAL DE LA INTEGRACIÓN**

### **1.1 Problemas Identificados**

#### **1.1.1 Fragmentación del Sistema**
El proyecto actual presenta una **arquitectura fragmentada** donde:

- **Wake Word**: Está implementado en `WakeWordProject/` como un sistema independiente
- **Pipeline Principal**: Está en `src/` con componentes separados
- **Main.py**: Usa un detector Mycroft que no corresponde al modelo entrenado
- **No hay Integración**: Los componentes no se comunican entre sí

#### **1.1.2 Inconsistencias de Implementación**
```python
# main.py actual - INCORRECTO
from src.wake_word import MycroftDetector  # ❌ No existe
detector = MycroftDetector("udito_model.net")  # ❌ Modelo incorrecto

# WakeWordProject - CORRECTO pero aislado
class HMMGMMProfessionalDetector:  # ✅ Modelo entrenado
    def __init__(self, model_path="udito_hmm_gmm_models.pth"):
        # Carga el modelo HMM/GMM real
```

#### **1.1.3 Configuración Desconectada**
- **`config/wake_word.json`**: Referencia a `udito_model.net` (inexistente)
- **`config/wake_word_config.json`**: Configuración para modelo diferente
- **Modelos entrenados**: Están en `WakeWordProject/` pero no se usan

### **1.2 Componentes Existentes No Integrados**

#### **1.2.1 Modelo Entrenado (HMM/GMM)**
- **Ubicación**: `WakeWordProject/udito_hmm_gmm_models.pth`
- **Arquitectura**: HMM + GMM con características MFCC temporales
- **Rendimiento**: 100% accuracy en test, threshold optimizado
- **Estado**: **FUNCIONANDO** pero **NO INTEGRADO**

#### **1.2.2 Detector Profesional**
- **Archivo**: `WakeWordProject/detector_hmm_gmm_professional.py`
- **Funcionalidades**: Detección en tiempo real, threshold adaptativo
- **Estado**: **FUNCIONANDO** pero **AISLADO**

#### **1.2.3 Pipeline de Audio**
- **Componente**: `src/voice/audio_handler_faster.py`
- **Funcionalidades**: Captura de audio, VAD, transcripción Whisper
- **Estado**: **FUNCIONANDO** pero **SIN WAKE WORD**

---

## **2. ARQUITECTURA DE INTEGRACIÓN CORRECTA**

### **2.1 Visión General de la Integración**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE UDI INTEGRADO                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │   WAKE WORD     │    │   AUDIO         │    │   TRANSCRIPCIÓN │        │
│  │   (HMM/GMM)     │───►│   HANDLER       │───►│   (Whisper)     │        │
│  │   INTEGRADO     │    │   UNIFICADO     │    │   INTEGRADO     │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│           │                       │                       │                 │
│           ▼                       ▼                       ▼                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │   CALLBACK      │    │   BUFFER        │    │   RAG           │        │
│  │   SYSTEM        │    │   MANAGER       │    │   PROCESSING    │        │
│  │   UNIFICADO     │    │   INTELIGENTE   │    │   INTEGRADO     │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### **2.2 Componentes de Integración Requeridos**

#### **2.2.1 Wake Word Integrado**
```python
# src/wake_word/hmm_gmm_detector.py
class HMMGMMWakeWordDetector:
    def __init__(self, config_path: str = "config/wake_word_config.json"):
        self.model = None
        self.config = self._load_config()
        self._load_trained_model()
        
    def _load_trained_model(self):
        """Carga el modelo HMM/GMM entrenado"""
        model_path = self.config['model']['path']
        self.model = joblib.load(model_path)
        
    def detect_wake_word(self, audio_chunk: np.ndarray) -> bool:
        """Detecta wake word en tiempo real"""
        # Implementación del detector HMM/GMM
        pass
```

#### **2.2.2 Audio Handler Unificado**
```python
# src/voice/unified_audio_handler.py
class UnifiedAudioHandler:
    def __init__(self, config_path: str = "config/settings.json"):
        self.wake_word_detector = HMMGMMWakeWordDetector()
        self.whisper_transcriber = WhisperTranscriber()
        self.audio_buffer = AudioBuffer()
        
    def start_listening(self):
        """Inicia escucha unificada: wake word + transcripción"""
        # Escucha continua para wake word
        # Cuando se detecta, activa transcripción
        pass
```

#### **2.2.3 Sistema de Callbacks Unificado**
```python
# src/core/callback_system.py
class UnifiedCallbackSystem:
    def __init__(self):
        self.on_wake_word_detected: Optional[Callable] = None
        self.on_speech_started: Optional[Callable] = None
        self.on_transcription_complete: Optional[Callable] = None
        self.on_response_ready: Optional[Callable] = None
        
    def wake_word_detected(self):
        """Callback cuando se detecta UDI"""
        if self.on_wake_word_detected:
            self.on_wake_word_detected()
```

---

## **3. IMPLEMENTACIÓN DE LA INTEGRACIÓN**

### **3.1 Estructura de Directorios Propuesta**

```
src/
├── wake_word/
│   ├── __init__.py
│   ├── hmm_gmm_detector.py          # Detector HMM/GMM integrado
│   └── wake_word_manager.py         # Gestor de wake word
├── voice/
│   ├── __init__.py
│   ├── unified_audio_handler.py     # Handler unificado
│   ├── audio_buffer.py              # Gestor de buffers
│   └── transcription_service.py     # Servicio de transcripción
├── core/
│   ├── __init__.py
│   ├── callback_system.py           # Sistema de callbacks
│   ├── pipeline_manager.py          # Gestor del pipeline
│   └── state_manager.py             # Gestor de estados
└── main.py                          # Punto de entrada unificado
```

### **3.2 Configuración Unificada**

#### **3.2.1 Configuración del Wake Word**
```json
// config/wake_word_config.json
{
    "wake_word": {
        "name": "UDI",
        "confidence_threshold": 0.4,
        "model_path": "WakeWordProject/udito_hmm_gmm_models.pth",
        "use_hmm": true,
        "adaptive_threshold": true
    },
    "audio": {
        "sample_rate": 16000,
        "chunk_size": 1024,
        "silence_threshold": 0.008,
        "min_silence_duration": 0.5,
        "max_word_duration": 1.5
    },
    "detection": {
        "continuous_listening": true,
        "pause_between_detections": 0.1,
        "cooldown_period": 2.0
    }
}
```

#### **3.2.2 Configuración del Pipeline**
```json
// config/pipeline_config.json
{
    "pipeline": {
        "wake_word_first": true,
        "transcription_after_wake": true,
        "rag_processing": true,
        "tts_response": true
    },
    "callbacks": {
        "enable_logging": true,
        "enable_metrics": true,
        "enable_debug": false
    }
}
```

### **3.3 Flujo de Integración**

#### **3.3.1 Secuencia de Activación**
```
1. ESCUCHA CONTINUA (Wake Word)
   ↓
2. DETECCIÓN "UDI" (HMM/GMM)
   ↓
3. ACTIVACIÓN DEL SISTEMA
   ↓
4. ESCUCHA CONSULTA (Whisper)
   ↓
5. TRANSCRIPCIÓN A TEXTO
   ↓
6. PROCESAMIENTO RAG
   ↓
7. GENERACIÓN RESPUESTA TTS
   ↓
8. RETORNO A ESCUCHA CONTINUA
```

#### **3.3.2 Estados del Sistema**
```python
class PipelineState:
    INACTIVE = "INACTIVE"           # Esperando wake word
    WAKE_WORD_DETECTED = "WAKE"    # UDI detectado
    LISTENING_QUERY = "LISTENING"  # Escuchando consulta
    PROCESSING = "PROCESSING"       # Procesando con RAG
    RESPONDING = "RESPONDING"       # Generando respuesta TTS
    RETURNING = "RETURNING"         # Volviendo a estado inicial
```

---

## **4. IMPLEMENTACIÓN TÉCNICA DETALLADA**

### **4.1 Integración del Detector HMM/GMM**

#### **4.1.1 Carga del Modelo Entrenado**
```python
def _load_trained_model(self):
    """Carga el modelo HMM/GMM entrenado desde WakeWordProject"""
    try:
        model_path = Path(self.config['model']['path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
            
        # Cargar modelo usando joblib
        self.model = joblib.load(model_path)
        
        # Extraer componentes del modelo
        self.udito_hmm = self.model['udito_hmm']
        self.not_udito_hmm = self.model['not_udito_hmm']
        self.udito_gmm = self.model['udito_gmm']
        self.not_udito_gmm = self.model['not_udito_gmm']
        self.threshold = self.model['threshold']
        
        self.logger.info(f"Modelo HMM/GMM cargado: threshold={self.threshold}")
        return True
        
    except Exception as e:
        self.logger.error(f"Error cargando modelo: {e}")
        return False
```

#### **4.1.2 Detección en Tiempo Real**
```python
def detect_wake_word(self, audio_chunk: np.ndarray) -> Tuple[bool, float, float]:
    """Detecta wake word usando el modelo entrenado"""
    try:
        # Extraer características según el modelo seleccionado
        if self.config['wake_word']['use_hmm']:
            features = self._extract_mfcc_temporal(audio_chunk)
            is_wake_word, probability, likelihood = self._detect_hmm(features)
        else:
            features = self._extract_mfcc_static(audio_chunk)
            is_wake_word, probability, likelihood = self._detect_gmm(features)
            
        # Aplicar threshold adaptativo si está habilitado
        if self.config['wake_word']['adaptive_threshold']:
            self._update_adaptive_threshold(probability)
            
        return is_wake_word, probability, likelihood
        
    except Exception as e:
        self.logger.error(f"Error en detección: {e}")
        return False, 0.0, 0.0
```

### **4.2 Handler de Audio Unificado**

#### **4.2.1 Gestión de Estados**
```python
class UnifiedAudioHandler:
    def __init__(self):
        self.state = PipelineState.INACTIVE
        self.wake_word_detector = HMMGMMWakeWordDetector()
        self.transcription_service = WhisperTranscriber()
        self.audio_buffer = AudioBuffer()
        
    def _handle_audio_chunk(self, audio_chunk: np.ndarray):
        """Maneja chunk de audio según el estado actual"""
        if self.state == PipelineState.INACTIVE:
            # Buscar wake word
            is_wake_word, prob, likelihood = self.wake_word_detector.detect_wake_word(audio_chunk)
            if is_wake_word:
                self._activate_system()
                
        elif self.state == PipelineState.LISTENING_QUERY:
            # Acumular audio para transcripción
            self.audio_buffer.add_chunk(audio_chunk)
            
        # ... otros estados
```

#### **4.2.2 Transiciones de Estado**
```python
def _activate_system(self):
    """Activa el sistema cuando se detecta wake word"""
    self.state = PipelineState.WAKE_WORD_DETECTED
    self.logger.info("🎉 ¡UDI detectado! Activando sistema...")
    
    # Notificar callbacks
    if self.callback_system.on_wake_word_detected:
        self.callback_system.on_wake_word_detected()
    
    # Cambiar a estado de escucha de consulta
    self.state = PipelineState.LISTENING_QUERY
    self.audio_buffer.clear()
    self.logger.info("🎤 Escuchando consulta...")
```

### **4.3 Sistema de Callbacks Unificado**

#### **4.3.1 Definición de Callbacks**
```python
class UnifiedCallbackSystem:
    def __init__(self):
        # Callbacks del wake word
        self.on_wake_word_detected: Optional[Callable] = None
        
        # Callbacks de audio
        self.on_speech_started: Optional[Callable] = None
        self.on_speech_ended: Optional[Callable] = None
        
        # Callbacks de transcripción
        self.on_transcription_started: Optional[Callable] = None
        self.on_transcription_complete: Optional[Callable] = None
        
        # Callbacks del pipeline
        self.on_rag_processing_started: Optional[Callable] = None
        self.on_response_ready: Optional[Callable] = None
        self.on_system_idle: Optional[Callable] = None
```

#### **4.3.2 Implementación de Callbacks**
```python
def _notify_wake_word_detected(self):
    """Notifica que se detectó el wake word"""
    if self.callback_system.on_wake_word_detected:
        try:
            self.callback_system.on_wake_word_detected()
        except Exception as e:
            self.logger.error(f"Error en callback wake word: {e}")

def _notify_transcription_complete(self, text: str):
    """Notifica que se completó la transcripción"""
    if self.callback_system.on_transcription_complete:
        try:
            self.callback_system.on_transcription_complete(text)
        except Exception as e:
            self.logger.error(f"Error en callback transcripción: {e}")
```

---

## **5. MAIN.PY INTEGRADO**

### **5.1 Estructura del Main Integrado**
```python
#!/usr/bin/env python3
"""
UDI - Sistema Principal Integrado
Pipeline completo: Wake Word → Transcripción → RAG → TTS
"""

import logging
import time
from src.core.pipeline_manager import PipelineManager
from src.core.callback_system import UnifiedCallbackSystem

def main():
    """Función principal del sistema UDI integrado"""
    print("🎧 UDI - Sistema de Asistente de Voz Integrado")
    print("🔊 Wake Word: UDI")
    print("🎯 Pipeline: Wake Word → STT → RAG → TTS")
    print("⏹️  Ctrl+C para salir")
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("UDI")
    
    try:
        # Crear sistema de callbacks
        callback_system = UnifiedCallbackSystem()
        
        # Configurar callbacks
        callback_system.on_wake_word_detected = on_wake_word_detected
        callback_system.on_transcription_complete = on_transcription_complete
        callback_system.on_response_ready = on_response_ready
        
        # Crear gestor del pipeline
        pipeline_manager = PipelineManager(callback_system)
        
        # Iniciar pipeline
        pipeline_manager.start()
        
        # Mantener sistema activo
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n👋 UDI detenido")
        if 'pipeline_manager' in locals():
            pipeline_manager.stop()
    except Exception as e:
        logger.error(f"Error en sistema principal: {e}")

def on_wake_word_detected():
    """Callback cuando se detecta UDI"""
    print("🎉 ¡UDI detectado!")
    print("🔔 Activando sistema...")

def on_transcription_complete(text: str):
    """Callback cuando se completa la transcripción"""
    print(f"📝 Transcripción: {text}")
    print("🧠 Procesando con RAG...")

def on_response_ready(response: str):
    """Callback cuando la respuesta está lista"""
    print(f"💬 Respuesta: {response}")
    print("🔊 Reproduciendo con TTS...")

if __name__ == "__main__":
    main()
```

---

## **6. MIGRACIÓN Y IMPLEMENTACIÓN**

### **6.1 Pasos de Migración**

#### **6.1.1 Fase 1: Preparación**
1. **Crear estructura de directorios** propuesta
2. **Mover modelo entrenado** a ubicación accesible
3. **Crear archivos de configuración** unificados
4. **Implementar detector HMM/GMM** integrado

#### **6.1.2 Fase 2: Integración**
1. **Implementar handler de audio** unificado
2. **Crear sistema de callbacks** unificado
3. **Integrar wake word** con transcripción
4. **Conectar con sistema RAG** existente

#### **6.1.3 Fase 3: Testing**
1. **Probar detección** de wake word
2. **Verificar flujo** completo del pipeline
3. **Optimizar rendimiento** y latencia
4. **Documentar** sistema integrado

### **6.2 Archivos a Crear/Modificar**

#### **6.2.1 Nuevos Archivos**
- `src/wake_word/hmm_gmm_detector.py`
- `src/wake_word/wake_word_manager.py`
- `src/voice/unified_audio_handler.py`
- `src/core/pipeline_manager.py`
- `src/core/callback_system.py`
- `src/core/state_manager.py`

#### **6.2.2 Archivos a Modificar**
- `main.py` → Integrar pipeline completo
- `config/settings.json` → Agregar configuración de wake word
- `src/voice/audio_handler_faster.py` → Integrar con wake word

#### **6.2.3 Archivos a Preservar**
- `WakeWordProject/` → Mantener para referencia y reentrenamiento
- `src/rag/` → Sistema RAG existente
- `src/tts/` → Sistema TTS existente

---

## **7. BENEFICIOS DE LA INTEGRACIÓN**

### **7.1 Beneficios Técnicos**
- **Sistema Unificado**: Un solo punto de entrada y control
- **Comunicación Eficiente**: Callbacks bien definidos entre componentes
- **Gestión de Estado**: Control centralizado del flujo del pipeline
- **Configuración Centralizada**: Un solo lugar para ajustar parámetros

### **7.2 Beneficios de Rendimiento**
- **Latencia Reducida**: Eliminación de overhead de comunicación entre procesos
- **Uso de Recursos Optimizado**: Compartir buffers y recursos de audio
- **Detección Más Precisa**: Integración directa entre wake word y transcripción
- **Recuperación Automática**: Manejo unificado de errores y fallos

### **7.3 Beneficios de Mantenimiento**
- **Código Centralizado**: Fácil debugging y mantenimiento
- **Testing Unificado**: Pruebas del pipeline completo
- **Documentación Integrada**: Un solo lugar para documentar el sistema
- **Escalabilidad**: Fácil agregar nuevos componentes

---

## **8. CONCLUSIONES Y RECOMENDACIONES**

### **8.1 Estado Actual**
El proyecto UDI tiene un **modelo de wake word excelente** (HMM/GMM con 100% accuracy) pero está **completamente aislado** del pipeline principal. Esto resulta en:

- **Funcionalidad limitada**: Solo wake word sin pipeline completo
- **Recursos desperdiciados**: Modelo entrenado no se utiliza
- **Arquitectura fragmentada**: Componentes separados sin comunicación

### **8.2 Solución Propuesta**
La **integración propuesta** crea un **sistema unificado** que:

- **Aprovecha el modelo entrenado**: HMM/GMM integrado al pipeline
- **Mantiene la arquitectura existente**: RAG y TTS sin cambios
- **Crea flujo coherente**: Wake word → STT → RAG → TTS
- **Permite evolución futura**: Fácil agregar nuevos componentes

### **8.3 Próximos Pasos**
1. **Implementar integración** siguiendo la arquitectura propuesta
2. **Migrar configuración** a archivos unificados
3. **Probar pipeline completo** con datos reales
4. **Optimizar rendimiento** y latencia
5. **Documentar sistema integrado** para uso futuro

---

## **9. CORRECCIÓN IMPORTANTE: SISTEMA DE ENTRENAMIENTO**

### **9.1 Cambio de Mycroft a Wakeword-Detector**

**⚠️ CORRECCIÓN CRÍTICA**: El sistema de entrenamiento se cambió completamente de **Mycroft Precise** a **wakeword-detector**.

#### **9.1.1 Sistema Anterior (INCORRECTO)**
- **Mycroft Precise**: Sistema obsoleto y problemático
- **Problemas**: Dependencias complejas, compatibilidad limitada
- **Estado**: **ABANDONADO** en favor de wakeword-detector

#### **9.1.2 Sistema Actual (CORRECTO)**
- **Wakeword-Detector**: Sistema moderno y eficiente
- **Ventajas**: Fácil instalación, mejor rendimiento, compatibilidad Windows
- **Implementación**: Completamente funcional en `WakeWordProject/`

### **9.2 Evidencia del Cambio**

#### **9.2.1 Archivos de Wakeword-Detector**
```
WakeWordProject/
├── env_wakeword_detector/          # Entorno virtual dedicado
├── extract_features_udito.py       # Extracción de características
├── train_hmm_gmm_professional.py  # Entrenamiento HMM/GMM
└── detector_hmm_gmm_professional.py # Detector en tiempo real
```

#### **9.2.2 Comandos de Wakeword-Detector**
```bash
# Comandos disponibles
wakeword-detector start      # Iniciar interfaz de grabación
wakeword-detector train      # Entrenar modelo
wakeword-detector extract    # Extraer características
wakeword-detector listen     # Escuchar en tiempo real
```

### **9.3 Implicaciones para la Integración**

#### **9.3.1 Modelos Entrenados**
- **Formato**: `.pth` (PyTorch/Joblib)
- **Ubicación**: `WakeWordProject/udito_hmm_gmm_models.pth`
- **Compatibilidad**: Total con Python estándar

#### **9.3.2 Dependencias**
- **Wakeword-Detector**: Solo para entrenamiento
- **Runtime**: Solo Python + Joblib + NumPy
- **Integración**: Sin dependencias externas complejas

---

**La integración del wake word HMM/GMM al pipeline UDI es esencial para aprovechar el excelente trabajo de entrenamiento realizado con wakeword-detector y crear un sistema de asistente de voz funcional y completo.**
