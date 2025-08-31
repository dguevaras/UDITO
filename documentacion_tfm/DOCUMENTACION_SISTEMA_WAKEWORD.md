# DOCUMENTACIÓN TÉCNICA DEL SISTEMA WAKE WORD UDI

## Resumen Ejecutivo

Este documento describe la implementación técnica del sistema de detección de palabra de activación (wake word) "UDITO" para el asistente de voz UDI. El sistema utiliza técnicas avanzadas de procesamiento de señales acústicas y aprendizaje profundo para proporcionar una detección robusta y eficiente.

## 1. Arquitectura del Sistema

### 1.1 Componentes Principales

- **Procesador de Audio**: Captura y preprocesamiento de señales acústicas
- **Extractor de Características MFCC**: Análisis espectral-temporal
- **Modelo de Clasificación**: Red neuronal convolucional TensorFlow Lite
- **Sistema de Confirmación**: Lógica de validación multi-hit
- **Interfaz de Usuario**: Sistema de confirmación y estadísticas

### 1.2 Flujo de Procesamiento

```
Audio Input → Preprocesamiento → MFCC → Modelo CNN → Clasificación → Confirmación → Activación
```

## 2. Implementación Técnica

### 2.1 Captura de Audio

**Parámetros de Configuración:**
- Frecuencia de muestreo: 16,000 Hz
- Resolución: 32-bit float
- Canales: Mono (1 canal)
- Tamaño de bloque: 1 segundo
- Solapamiento: 50% entre bloques

**Implementación:**
```python
with sd.InputStream(
    samplerate=CONFIG['sample_rate'],
    channels=1,
    dtype=np.float32,
    blocksize=int(CONFIG['sample_rate'] * 0.1),
    callback=callback
)
```

### 2.2 Extracción de Características MFCC

**Parámetros de Análisis:**
- Ventana de análisis: 25ms
- Paso entre ventanas: 10ms
- Número de coeficientes MFCC: 13
- Bins de frecuencia mel: 40
- Longitud FFT: 512 puntos

**Implementación Matemática:**
1. **Transformada de Fourier de Tiempo Corto (STFT):**
   ```
   STFT(x[n]) = Σ x[n]w[n-m]e^(-j2πkn/N)
   ```

2. **Espectrograma de Mel:**
   ```
   Mel(f) = 2595 log₁₀(1 + f/700)
   ```

3. **Coeficientes MFCC:**
   ```
   MFCC[k] = Σ log(S[m]) cos(πk(m+0.5)/M)
   ```

**Código de Implementación:**
```python
def extraer_mfcc(audio, sample_rate=16000, num_mfcc=13, 
                  frame_length_ms=25, frame_step_ms=10):
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)
    frame_length = int(sample_rate * frame_length_ms / 1000)
    frame_step = int(sample_rate * frame_step_ms / 1000)
    
    stft = tf.signal.stft(audio, frame_length=frame_length, 
                          frame_step=frame_step, fft_length=512)
    spectrogram = tf.abs(stft)
    
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40, num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=sample_rate
    )
    mel_spectrogram = tf.tensordot(spectrogram, mel_w, 1)
    log_mel = tf.math.log(mel_spectrogram + 1e-6)
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :num_mfcc]
    return mfcc.numpy()
```

### 2.3 Modelo de Clasificación

**Arquitectura de la Red Neuronal:**
- **Forma de entrada**: [1, 13, 100, 1]
  - Batch size: 1
  - Altura: 13 coeficientes MFCC
  - Ancho: 100 frames temporales
  - Canales: 1 (escala de grises)
- **Forma de salida**: [1, 1]
  - Probabilidad de activación: 0.0 - 1.0

**Implementación TensorFlow Lite:**
```python
interpreter = tf.lite.Interpreter(model_path=CONFIG['model_path'])
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preparación de entrada
x = mfcc.T.reshape(1, 13, 100, 1)
interpreter.set_tensor(input_details[0]['index'], x.astype(np.float32))
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
prob = float(output[0][0])
```

### 2.4 Sistema de Confirmación Multi-Hit

**Lógica de Validación:**
El sistema implementa un mecanismo de confirmación doble para evitar falsos positivos:

1. **Primera Detección (Hit 1/2):**
   - Probabilidad > umbral (49%)
   - Inicia contador de hits consecutivos

2. **Segunda Detección (Hit 2/2):**
   - Probabilidad > umbral en bloque consecutivo
   - Confirma activación del wake word

3. **Período de Cooldown:**
   - Duración: 0.5 segundos
   - Propósito: Evitar activaciones múltiples
   - Comportamiento: Ignora detecciones durante este período

**Implementación:**
```python
if avg_prob >= CONFIG['wakeword_threshold']:
    consecutive_hits += 1
    if consecutive_hits >= 2:  # Confirmación doble
        if current_time - last_detection > CONFIG['cool_down_sec']:
            # Activar sistema
            self.activar_sistema(avg_prob)
            last_detection = current_time
            consecutive_hits = 0
```

## 3. Parámetros de Configuración

### 3.1 Umbrales de Detección

- **Umbral de Voz**: 0.0002 (energía mínima para considerar actividad vocal)
- **Umbral de Wake Word**: 0.49 (probabilidad mínima para clasificar como UDITO)
- **Cooldown**: 0.5 segundos (tiempo de bloqueo post-activación)

### 3.2 Parámetros de Procesamiento

- **Tamaño de Buffer**: 5 muestras para promedio móvil
- **Histéresis**: 0.10 (diferencia entre umbrales de activación/desactivación)
- **Sensibilidad**: 0.6 (balance entre precisión y sensibilidad)

## 4. Análisis de Rendimiento

### 4.1 Métricas de Evaluación

- **Latencia de Respuesta**: < 100ms (objetivo)
- **Precisión**: Basada en confirmación del usuario
- **Falsos Positivos**: Minimizados por sistema multi-hit
- **Falsos Negativos**: Reducidos por umbral adaptativo

### 4.2 Optimizaciones Implementadas

1. **Procesamiento Selectivo**: Solo analiza bloques con actividad vocal
2. **Reducción de FFT**: 512 puntos en lugar de 1024
3. **Promedio Móvil**: Suavizado de probabilidades para estabilidad
4. **Cooldown Inteligente**: Evita procesamiento redundante

## 5. Características de Robustez

### 5.1 Filtrado de Ruido

- **Detección de Actividad Vocal (VAD)**: Basada en energía del señal
- **Normalización RMS**: Compensación de variaciones de volumen
- **Filtrado Temporal**: Promedio móvil de probabilidades

### 5.2 Adaptabilidad

- **Umbrales Dinámicos**: Ajuste automático según condiciones ambientales
- **Calibración Continua**: Actualización de línea base en tiempo real
- **Recuperación de Errores**: Reinicio automático tras fallos

## 6. Interfaz de Usuario

### 6.1 Sistema de Confirmación

El usuario confirma manualmente cada detección para:
- **Validar Precisión**: Confirmar detecciones correctas
- **Registrar Falsos Positivos**: Mejorar entrenamiento del modelo
- **Control de Calidad**: Mantener estándares de rendimiento

### 6.2 Feedback Visual

- **Indicadores de Estado**: Escuchando, Procesando, Detectado
- **Información de Probabilidad**: Nivel de confianza de cada detección
- **Estadísticas en Tiempo Real**: Conteo de activaciones y precisión

## 7. Limitaciones y Consideraciones

### 7.1 Limitaciones Técnicas

- **Dependencia del Usuario**: Requiere confirmación manual
- **Sensibilidad Ambiental**: Afectado por ruido de fondo
- **Latencia de Procesamiento**: Limitada por complejidad del modelo

### 7.2 Consideraciones de Implementación

- **Compatibilidad de Plataforma**: Optimizado para Windows
- **Recursos del Sistema**: Uso moderado de CPU y memoria
- **Escalabilidad**: Arquitectura modular para futuras mejoras

## 8. Conclusiones

El sistema de wake word implementado proporciona una base sólida para la activación por voz del asistente UDI. La combinación de técnicas avanzadas de procesamiento de señales, aprendizaje profundo y validación multi-hit resulta en un sistema robusto y confiable.

### 8.1 Logros Principales

- **Detección Robusta**: Sistema multi-hit para alta precisión
- **Interfaz Intuitiva**: Confirmación manual para control de calidad
- **Rendimiento Optimizado**: Latencia < 100ms objetivo
- **Arquitectura Modular**: Fácil mantenimiento y extensión

### 8.2 Direcciones Futuras

- **Entrenamiento Continuo**: Mejora del modelo con datos del usuario
- **Adaptación Acústica**: Calibración automática del entorno
- **Integración Completa**: Conexión con pipeline STT-NLP-TTS
- **Optimización de Plataforma**: Soporte para múltiples sistemas operativos

---

**Documento Técnico - Sistema Wake Word UDI**  
**Versión**: 1.0  
**Fecha**: Agosto 2025  
**Autor**: Sistema UDI  
**Clasificación**: Técnica Interna
