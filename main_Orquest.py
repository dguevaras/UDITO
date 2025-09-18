#!/usr/bin/env python3
"""
Orquestador Principal del Sistema UDI - VERSIÓN SIMPLIFICADA
Integra Wake Word -> STT -> RAG -> TTS usando la lógica probada de main_wakeword.py
"""

import os
import sys
import json
import time
import threading
import tempfile
import queue
from pathlib import Path

# Configurar variables de entorno para TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import sounddevice as sd
import tensorflow as tf
from collections import deque
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UDI_Orchestrator")

# Configuración (copiada de main_wakeword.py)
CONFIG = {
    "model_path": "WakeWord-project/micro_model.tflite",
    "sample_rate": 16000,
    "block_duration": 1.0,
    "threshold_voice": 0.0002,
    "wakeword_threshold": 0.49,
    "chunk_ms": 80,
    "sensitivity": 0.6,
    "calibrate_secs": 3.0,
    "cool_down_sec": 3.0  # Aumentar cooldown para evitar ciclos
}

class UDIOrchestrator:
    """Orquestador principal del sistema UDI"""
    
    def __init__(self):
        """Inicializa el orquestador"""
        self.is_running = False
        self.is_listening = False
        self.last_activation = 0
        self.stt_start_delay_sec = 1.5
        self.stt_config = None
        self.rag_config = None
        self.tts_playing = False
        
        # Componentes del sistema
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        # Cola de audio
        self.audio_queue = queue.Queue()
        
        # Cargar configuraciones externas
        try:
            with open("config/wake_word_config.json", "r", encoding="utf-8") as f:
                ww_cfg = json.load(f)
            model_path = ww_cfg.get("model", {}).get("path")
            if model_path:
                CONFIG["model_path"] = model_path
            audio = ww_cfg.get("audio", {})
            if "sample_rate" in audio:
                CONFIG["sample_rate"] = audio["sample_rate"]
            processing = ww_cfg.get("processing", {})
            if "block_duration_sec" in processing:
                CONFIG["block_duration"] = processing["block_duration_sec"]
            detection = ww_cfg.get("detection", {})
            for k_json, k_conf in [("threshold_voice","threshold_voice"),("wakeword_threshold","wakeword_threshold"),("sensitivity","sensitivity"),("cool_down_sec","cool_down_sec")]:
                if k_json in detection:
                    CONFIG[k_conf] = detection[k_json]
        except Exception as e:
            logger.error(f"Error cargando wake word config: {e}")

        try:
            with open("config/STT_config.json", "r", encoding="utf-8") as f:
                self.stt_config = json.load(f)
            self.stt_start_delay_sec = float(self.stt_config.get("audio_capture", {}).get("start_delay_sec", self.stt_start_delay_sec))
        except Exception as e:
            logger.error(f"Error cargando STT config: {e}")

        try:
            with open("config/rag_config.json", "r", encoding="utf-8") as f:
                self.rag_config = json.load(f)
        except Exception:
            self.rag_config = None

        logger.info("Orquestador UDI inicializado")
    
    def cargar_modelo_wakeword(self):
        """Cargar modelo TensorFlow Lite (copiado de main_wakeword.py)"""
        logger.info("Cargando modelo TensorFlow Lite...")
        try:
            self.interpreter = tf.lite.Interpreter(model_path=CONFIG['model_path'])
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logger.info(f"Modelo cargado exitosamente")
            logger.info(f"Forma de entrada: {self.input_details[0]['shape']}")
            logger.info(f"Forma de salida: {self.output_details[0]['shape']}")
            # Guardar rango de entrada para adaptar la inferencia
            try:
                self._model_input_rank = len(self.input_details[0]['shape'])
            except Exception:
                self._model_input_rank = 2
            return True
        except Exception as e:
            logger.error(f"Error al cargar modelo: {e}")
            return False
    
    def extraer_mfcc(self, audio, sample_rate=16000, num_mfcc=13, frame_length_ms=25, frame_step_ms=10):
        """Extraer características MFCC del audio (copiado de main_wakeword.py)"""
        try:
            audio = tf.convert_to_tensor(audio, dtype=tf.float32)
            frame_length = int(sample_rate * frame_length_ms / 1000)
            frame_step = int(sample_rate * frame_step_ms / 1000)
            
            stft = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=512)
            spectrogram = tf.abs(stft)
            
            num_spectrogram_bins = spectrogram.shape[-1]
            mel_w = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=40,
                num_spectrogram_bins=num_spectrogram_bins,
                sample_rate=sample_rate
            )
            mel_spectrogram = tf.tensordot(spectrogram, mel_w, 1)
            log_mel = tf.math.log(mel_spectrogram + 1e-6)
            mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :num_mfcc]
            return mfcc.numpy()
        except Exception as e:
            logger.error(f"Error extracción MFCC: {e}")
            return None
    
    def ejecutar_inferencia(self, bloque_audio):
        """Ejecutar inferencia en bloque de audio (copiado de main_wakeword.py)"""
        try:
            input_shape = self.input_details[0]['shape']
            input_rank = getattr(self, '_model_input_rank', len(input_shape))

            if input_rank == 4:
                # Ruta MFCC [1,13,100,1]
                mfcc = self.extraer_mfcc(bloque_audio)
                if mfcc is None:
                    return 0.5
                if mfcc.shape[0] > 100:
                    mfcc = mfcc[:100, :]
                elif mfcc.shape[0] < 100:
                    padding = np.zeros((100 - mfcc.shape[0], 13))
                    mfcc = np.vstack([mfcc, padding])
                x = mfcc.T.reshape(1, 13, 100, 1).astype(np.float32)
            else:
                # Ruta escalar [1,1] (usar energía RMS normalizada)
                rms = float(np.sqrt(np.mean(np.square(bloque_audio))))
                # Normalizar energía a un rango razonable
                x = np.array([[rms]], dtype=np.float32)

            self.interpreter.set_tensor(self.input_details[0]['index'], x)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Si la salida es [1,2] tomar prob de clase positiva si existe
            try:
                if output.shape[-1] == 2:
                    prob = float(output[0][1])
                else:
                    prob = float(output.flatten()[0])
            except Exception:
                prob = 0.5
            return prob
        except Exception as e:
            logger.error(f"Error inferencia: {e}")
            return 0.5
    
    def _handle_wakeword_detection(self):
        """Maneja la detección de wake word"""
        current_time = time.time()
        
        # Verificar cooldown y que TTS no esté hablando
        if current_time - self.last_activation < CONFIG['cool_down_sec'] or self.tts_playing:
            return
        
        self.last_activation = current_time
        logger.info("🎯 Wake word 'UDITO' detectado!")
        
        # Saludar con TTS
        self._speak_greeting()
        
        # Activar STT (esperar un poco para no capturar el saludo)
        try:
            time.sleep(self.stt_start_delay_sec)
        except Exception:
            pass
        self._activate_stt()
    
    def _speak_greeting(self):
        """Reproduce saludo inicial"""
        try:
            greeting_text = "Hola, en qué te puedo ayudar hoy"
            logger.info(f"🔊 Reproduciendo: {greeting_text}")
            
            # Reproducir en hilo separado para no bloquear
            def speak_thread():
                try:
                    self.tts_playing = True
                    self._speak_with_piper(greeting_text)
                finally:
                    self.tts_playing = False
            
            threading.Thread(target=speak_thread, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Error reproduciendo saludo: {e}")
    
    def _speak_with_piper(self, text):
        """Reproduce texto con Piper TTS"""
        try:
            import subprocess
            import tempfile
            
            # Crear archivos temporales
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w', encoding='utf-8') as temp_text:
                temp_text.write(text)
                temp_text_path = temp_text.name
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # Comando para Piper
            piper_path = Path("piper/piper.exe")
            voice_path = Path("piper/voices/es_ES-sharvard-medium.onnx")
            
            if piper_path.exists() and voice_path.exists():
                cmd = [
                    str(piper_path),
                    '--model', str(voice_path),
                    '--output_file', temp_audio_path
                ]
                
                # Ejecutar Piper
                with open(temp_text_path, 'r', encoding='utf-8') as input_file:
                    process = subprocess.Popen(
                        cmd,
                        stdin=input_file,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    process.communicate()
                
                # Reproducir audio
                if os.path.exists(temp_audio_path):
                    import pygame
                    pygame.mixer.init()
                    pygame.mixer.music.load(temp_audio_path)
                    pygame.mixer.music.play()
                    
                    # Esperar a que termine
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                
                # Limpiar archivos temporales
                try:
                    os.unlink(temp_text_path)
                    os.unlink(temp_audio_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error reproduciendo con Piper: {e}")
    
    def _activate_stt(self):
        """Activa el sistema STT para capturar instrucción"""
        try:
            logger.info("🎤 Activando STT - Escuchando instrucción...")
            
            # Configuración de grabación desde STT_config
            stt_audio = (self.stt_config or {}).get("audio_capture", {})
            SAMPLE_RATE = int(stt_audio.get("sample_rate", 16000))
            RECORD_DURATION = float(stt_audio.get("record_duration_sec", 5.0))
            RECORD_SAMPLES = int(SAMPLE_RATE * RECORD_DURATION)
            
            # Grabar audio
            audio_data = sd.rec(RECORD_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()  # Esperar a que termine la grabación
            
            # Convertir a int16
            audio_int16 = (audio_data * 32767).astype(np.int16)

            # EXTENSIÓN: si hay voz al final, extender 1.0 s más y concatenar antes de escribir
            try:
                tail_rms = float(np.sqrt(np.mean(np.square(audio_int16[-int(0.5*SAMPLE_RATE):].astype(np.float32)))))
                if tail_rms > 200:  # umbral simple
                    extra_samples = int(SAMPLE_RATE * 1.0)
                    extra = sd.rec(extra_samples, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
                    sd.wait()
                    extra_i16 = (extra * 32767).astype(np.int16)
                    audio_int16 = np.concatenate([audio_int16, extra_i16.reshape(-1)])
                    logger.info("Extensión de grabación aplicada (voz detectada al final)")
            except Exception as _:
                pass

            # Guardar en archivo temporal (una sola vez)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            import soundfile as sf
            sf.write(temp_path, audio_int16, SAMPLE_RATE)

            # Transcribir con Faster-Whisper
            logger.info("📝 Transcribiendo audio...")
            transcription_result = self._transcribe_with_faster_whisper(temp_path)
            
            # Limpiar archivo temporal
            try:
                os.unlink(temp_path)
            except:
                pass
            
            if transcription_result and transcription_result.get('text'):
                user_text = transcription_result['text'].strip()
                lower_text = user_text.lower()
                # Evitar procesar nuestras propias frases
                forbidden = [
                    'en qué te puedo ayudar hoy',
                    'entendí que preguntaste',
                    'podrías ser más específico'
                ]
                if any(p in lower_text for p in forbidden):
                    logger.info("Se ignoró texto generado por el sistema para evitar ciclo")
                    return
                logger.info(f"👤 Usuario dijo: {user_text}")
                
                # Procesar con RAG
                self._process_with_rag(user_text)
            else:
                logger.warning("No se detectó habla en el audio")
                self._speak_error("No pude entender lo que dijiste")
                
        except Exception as e:
            logger.error(f"Error en STT: {e}")
            self._speak_error("Tuve un problema al escucharte")
    
    def _transcribe_with_faster_whisper(self, audio_path):
        """Transcribe audio usando Faster-Whisper"""
        try:
            from faster_whisper import WhisperModel
            
            # Cargar modelo desde STT_config
            stt = (self.stt_config or {}).get("stt", {})
            model_name = stt.get("model_name", "small")
            device = stt.get("device", "cpu")
            compute_type = stt.get("compute_type", "int8")
            beam_size = int(stt.get("beam_size", 5))
            language = stt.get("language", "es")
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            
            # Transcribir
            segments, info = model.transcribe(audio_path, beam_size=beam_size, language=language)
            
            # Recopilar resultados
            transcription = ""
            for segment in segments:
                transcription += segment.text + " "
            
            if transcription.strip():
                return {
                    "text": transcription.strip(),
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "duration": info.duration
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error transcribiendo con Faster-Whisper: {e}")
            return None
    
    def _process_with_rag(self, user_text):
        """Procesa la consulta del usuario con RAG"""
        try:
            logger.info(f"🧠 Procesando consulta: {user_text}")
            
            # Verificar comandos de despedida
            despedida_keywords = ["nada", "adiós", "adios", "hasta luego", "cancela", "cancelar", "salir", "terminar", "fin"]
            if any(keyword in user_text.lower() for keyword in despedida_keywords):
                logger.info("👋 Comando de despedida detectado")
                answer = "Fue un placer servirte"
                self._speak_response(answer)
                logger.info("ESTADO: Finalizando activación - Volviendo al modo escucha de wake word")
                return
            
            # Respuesta simple por ahora
            if "horario" in user_text.lower():
                answer = "Los horarios de la universidad son de 8:00 a 18:00 de lunes a viernes."
            elif "matrícula" in user_text.lower():
                answer = "Para información sobre matrícula, puedes consultar la secretaría académica."
            elif "universidad" in user_text.lower():
                answer = "La UDIT es la Universidad de Diseño, Innovación y Tecnología."
            else:
                answer = f"Entendí que preguntaste sobre: {user_text}. ¿Podrías ser más específico?"
            
            logger.info(f"🤖 Respuesta: {answer}")
            
            # Reproducir respuesta con TTS
            self._speak_response(answer)
            logger.info("ESTADO: Finalizando activación - Volviendo al modo escucha de wake word")
                
        except Exception as e:
            logger.error(f"Error procesando con RAG: {e}")
            self._speak_error("Tuve un problema al procesar tu consulta")
            logger.info("ESTADO: Finalizando activación - Volviendo al modo escucha de wake word")
    
    def _speak_response(self, text):
        """Reproduce respuesta con TTS"""
        try:
            logger.info(f"🔊 Reproduciendo respuesta: {text[:50]}...")
            
            def speak_thread():
                try:
                    self.tts_playing = True
                    self._speak_with_piper(text)
                finally:
                    self.tts_playing = False
                # Después de reproducir, esperar un poco antes de volver al modo escucha
                time.sleep(2.0)  # Esperar 2 segundos adicionales
                logger.info("ESTADO: Volviendo al modo escucha de wake word")
            
            threading.Thread(target=speak_thread, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Error reproduciendo respuesta: {e}")
    
    def _speak_error(self, error_text):
        """Reproduce mensaje de error"""
        try:
            def speak_thread():
                try:
                    self.tts_playing = True
                    self._speak_with_piper(error_text)
                finally:
                    self.tts_playing = False
                # Después de reproducir error, esperar un poco antes de volver al modo escucha
                time.sleep(2.0)  # Esperar 2 segundos adicionales
                logger.info("ESTADO: Volviendo al modo escucha de wake word")
            
            threading.Thread(target=speak_thread, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Error reproduciendo mensaje de error: {e}")
    
    def ejecutar_sistema(self):
        """Bucle principal del sistema (basado en main_wakeword.py)"""
        if not self.cargar_modelo_wakeword():
            return
            
        logger.info("Iniciando stream de audio...")
        logger.info("🎧 Escuchando wake word 'UDITO'...")
        
        audio_q = queue.Queue()
        block_samples = int(CONFIG['sample_rate'] * CONFIG['block_duration'])
        
        def callback(indata, frames, ctime, status):
            if status:
                logger.warning(f"Estado audio: {status}")
            audio_q.put(indata.copy().reshape(-1))
            
        try:
            with sd.InputStream(
                samplerate=CONFIG['sample_rate'],
                channels=1,
                dtype=np.float32,
                blocksize=int(CONFIG['sample_rate'] * 0.1),
                callback=callback
            ):
                logger.info("Sistema listo - Escuchando wake word")
                logger.info("Di 'UDITO' para activar el sistema")
                
                buffer = np.zeros(0, dtype=np.float32)
                probs_hist = deque(maxlen=5)
                consecutive_hits = 0
                last_detection = 0
                
                while True:
                    try:
                        data = audio_q.get(timeout=1.0)
                        buffer = np.concatenate([buffer, data])
                        
                        while buffer.size >= block_samples:
                            block = buffer[:block_samples]
                            buffer = buffer[block_samples:]
                            
                            # Verificar actividad de voz
                            energy = float(np.mean(block ** 2))
                            if energy > CONFIG['threshold_voice']:
                                logger.info("AUDIO: Voz detectada - Procesando...")
                                
                                # Ejecutar inferencia
                                prob = self.ejecutar_inferencia(block)
                                probs_hist.append(prob)
                                avg_prob = float(np.mean(probs_hist))
                                
                                logger.info(f"WAKE: Prob={avg_prob:.3f}")
                                
                                # Detección de wake word (usar umbral de main_wakeword.py)
                                if avg_prob >= CONFIG['wakeword_threshold']:
                                    consecutive_hits += 1
                                    logger.info(f"WAKE: Hit {consecutive_hits}/2")
                                    if consecutive_hits >= 2:
                                        current_time = time.time()
                                        if current_time - last_detection > CONFIG['cool_down_sec']:
                                            self._handle_wakeword_detection()
                                            last_detection = current_time
                                            consecutive_hits = 0
                                            probs_hist.clear()
                                            logger.info("ESTADO: Continuando escucha...")
                                else:
                                    consecutive_hits = 0
                            else:
                                # Mostrar que está escuchando (cada 5 segundos)
                                current_time = time.time()
                                if not hasattr(self, 'last_listening_msg') or current_time - getattr(self, 'last_listening_msg', 0) > 5:
                                    logger.info("ESTADO: Escuchando...")
                                    self.last_listening_msg = current_time
                                        
                    except queue.Empty:
                        continue
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        logger.error(f"Error de procesamiento: {e}")
                        continue
                        
        except KeyboardInterrupt:
            logger.info("Interrupción recibida - Cerrando...")
        except Exception as e:
            logger.error(f"Error en stream de audio: {e}")
        finally:
            logger.info("Sistema UDI cerrado")
    
    def run(self):
        """Ejecuta el orquestador principal"""
        try:
            logger.info("🚀 Iniciando Sistema UDI")
            logger.info("=" * 50)
            
            logger.info("✅ Sistema UDI listo")
            logger.info("Presiona Ctrl+C para salir")
            logger.info("=" * 50)
            
            # Iniciar detección de wake word
            self.is_running = True
            self.ejecutar_sistema()
            
        except KeyboardInterrupt:
            logger.info("🛑 Sistema detenido por usuario")
        except Exception as e:
            logger.error(f"Error en sistema principal: {e}")
        finally:
            self.is_running = False
            logger.info("👋 Sistema UDI cerrado")

def main():
    """Función principal"""
    orchestrator = UDIOrchestrator()
    orchestrator.run()

if __name__ == "__main__":
    main()