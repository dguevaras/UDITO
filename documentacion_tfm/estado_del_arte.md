
# Estado del Arte

Este capítulo presenta el panorama actual sobre tecnologías clave utilizadas en asistentes de voz, alineado a las fases del proyecto: WakeWord, STT, RAG y TTS. Incluye comparativas técnicas, referencias actualizadas y destaca aportaciones específicas del proyecto.

---

## 2.1 WakeWord (Detección de palabra de activación)

Los sistemas WakeWord permiten pasar de un modo de escucha pasiva a activa mediante una palabra clave. Existen motores optimizados para dispositivos de borde como Porcupine (Picovoice), detectores entrenables como wakeword-detector, y soluciones obsoletas como Snowboy.

| Motor               | Latencia | Entrenamiento | Comentarios                    |
|---------------------|---------|----------------|--------------------------------|
| Porcupine           | Muy baja| Alta (web UI)  | Ligero y optimizado para edge  |
| wakeword-detector   | Media   | Requiere dataset| Reentrenable y open source     |
| Enfoque híbrido     | Variable| Interno        | Balance entre NN y matching    |

---

## 2.2 STT — Faster-Whisper y alternativas

Se eligió *Faster‑Whisper*, una versión optimizada del modelo Whisper de OpenAI. Es hasta cuatro veces más rápida, soporta cuantización *int8* y callbacks para integración eficiente en pipelines.

| Modelo               | Offline | Velocidad       | Uso             | Notas                                        |
|----------------------|--------|-----------------|------------------|----------------------------------------------|
| Faster‑Whisper       | Sí     | Alta (2–4×)     | Local            | Ideal para edge con optimización *int8*      |
| Whisper original     | Sí     | Media           | Local            | Preciso, pero lento                          |
| VOSK                 | Sí     | Alta            | Edge ligero      | Muy rápido y ligero                          |
| Servicios Cloud      | No     | Muy baja        | Cloud            | Alta precisión, pero costosos y con latencia |

---

## 2.3 RAG (Retrieval-Augmented Generation)

RAG combina recuperación semántica de información con generación de respuestas contextualizadas. Utiliza FAISS, ChromaDB o Milvus como vector stores, junto a frameworks como LangChain.

---

## 2.4 TTS — Piper y alternativas

Piper fue seleccionado por su alta calidad de voz, soporte ONNX, fonetización con eSpeak-NG, y capacidad para expresar emociones.

| Sistema         | Local | Naturalidad  | Emo/Prosodia | Comentarios                                |
|----------------|------|--------------|--------------|---------------------------------------------|
| Piper (ONNX)   | Sí   | Alta         | Sí           | Ideal para edge, fonetización y variabilidad |
| Coqui / VITS   | Sí   | Alta         | Sí           | Flexible, pero más pesado                  |
| ElevenLabs     | No   | Muy alta     | Sí           | Calidad top, pero solo cloud                |
| Amazon Polly   | No   | Alta         | Limitada     | Económico, con buen control de salida       |

---

## 2.5 Identificación de hablantes

Speaker ID permite identificar quién habla para personalizar interacción y permisos. Herramientas como pyannote.audio, Resemblyzer y ECAPA-TDNN son estándar.

---

## 2.6 Brechas identificadas

Este proyecto aporta un sistema integrado y ejecutable en edge que une:
- Detección híbrida de WakeWord
- STT offline optimizado (Faster‑Whisper)
- RAG local con vector store y QA generativo
- TTS emocional con Piper ONNX
