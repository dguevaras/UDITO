#!/usr/bin/env python3
"""
UDI - Main Simple WakeWord + Piper TTS
Sistema básico de activación por voz con respuesta TTS
"""

# Configurar TensorFlow para evitar colgadas - DEBE IR ANTES DE CUALQUIER IMPORT
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Solo errores críticos
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Deshabilitar optimizaciones problemáticas

import time
import queue
import numpy as np
import sounddevice as sd
import tensorflow as tf
from collections import deque
import subprocess
import sys
import threading

# Configuración
CONFIG = {
    'sample_rate': 16000,
    'block_duration': 1.0,
    'threshold_voice': 0.0002,  # Aumentado para ser más estricto
    'wakeword_threshold': 0.45,  # valores entre my estricto y poco estrico 0.1 0.9
    'cool_down_sec': 2.0,  # Mantener cooldown
    'model_path': 'WakeWord-project/wakeword_model.tflite'  # Ruta correcta
}

class SimpleWakeWordSystem:
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.last_detection = 0
        
    def cargar_modelo(self):
        """Cargar modelo TensorFlow Lite"""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=CONFIG['model_path'])
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            return True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False
            
    def extraer_mfcc(self, audio, sample_rate=16000, num_mfcc=13, frame_length_ms=25, frame_step_ms=10):
        """Extraer características MFCC del audio"""
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
        except Exception:
            return None
            
    def ejecutar_inferencia(self, bloque_audio):
        """Ejecutar inferencia en bloque de audio"""
        try:
            mfcc = self.extraer_mfcc(bloque_audio)
            if mfcc is None:
                return 0.5
                
            # Preparar entrada para modelo [1, 13, 100, 1]
            if mfcc.shape[0] > 100:
                mfcc = mfcc[:100, :]
            elif mfcc.shape[0] < 100:
                padding = np.zeros((100 - mfcc.shape[0], 13))
                mfcc = np.vstack([mfcc, padding])
                
            x = mfcc.T.reshape(1, 13, 100, 1)
            
            self.interpreter.set_tensor(self.input_details[0]['index'], x.astype(np.float32))
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            return float(output[0][0])
        except Exception:
            return 0.5
            
    def hablar_con_piper(self, texto):
        """Hacer hablar a Piper TTS usando la implementación existente"""
        def _hablar():
            try:
                # Ocultar mensajes de pygame
                import os
                os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
                
                # Importar el TTS existente
                import sys
                sys.path.append('src')
                from tts.piper_tts_real import PiperTTS
                
                # Crear instancia de Piper TTS
                tts = PiperTTS()
                
                # Generar y reproducir audio
                tts.speak(texto)
                
            except Exception as e:
                print(f"Error reproduciendo audio: {e}")
        
        # Ejecutar en thread separado para no bloquear
        thread = threading.Thread(target=_hablar)
        thread.daemon = True
        thread.start()
            
    def ejecutar_sistema(self):
        """Ejecutar sistema completo"""
        if not self.cargar_modelo():
            return
            
        print("Sistema iniciado")
        print("Escuchando...")
        
        audio_q = queue.Queue()
        block_samples = int(CONFIG['sample_rate'] * CONFIG['block_duration'])
        
        def callback(indata, frames, ctime, status):
            audio_q.put(indata.copy().reshape(-1))
            
        try:
            with sd.InputStream(
                samplerate=CONFIG['sample_rate'],
                channels=1,
                dtype=np.float32,
                blocksize=int(CONFIG['sample_rate'] * 0.1),
                callback=callback
            ):
                buffer = np.zeros(0, dtype=np.float32)
                probs_hist = deque(maxlen=5)
                consecutive_hits = 0
                
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
                                # Ejecutar inferencia
                                prob = self.ejecutar_inferencia(block)
                                probs_hist.append(prob)
                                avg_prob = float(np.mean(probs_hist))
                                
                                # Debug para probabilidades altas
                                if avg_prob > 0.5:  # Solo mostrar probabilidades altas
                                    print(f"Prob: {avg_prob:.3f}, Energy: {energy:.6f}")
                                
                                # Detectar wakeword con threshold alto
                                if avg_prob >= CONFIG['wakeword_threshold']:
                                    consecutive_hits += 1
                                    print(f"Hit {consecutive_hits}/2")
                                    if consecutive_hits >= 2:  # Confirmación doble
                                        current_time = time.time()
                                        if current_time - self.last_detection > CONFIG['cool_down_sec']:
                                            print("\nActivado")
                                            self.hablar_con_piper("Hola, en qué te puedo ayudar?")
                                            self.last_detection = current_time
                                            consecutive_hits = 0
                                            probs_hist.clear()
                                            print("--- Cooldown activo ---")
                                else:
                                    consecutive_hits = 0
                                    # Debug de ruido detectado
                                    if avg_prob > 0.2:  # Mostrar ruido con probabilidad moderada
                                        print(f"Ruido detectado - Prob: {avg_prob:.3f}, Energy: {energy:.6f}")
                            else:
                                # Debug de ruido de fondo
                                if energy > CONFIG['threshold_voice'] * 0.5:  # Ruido de fondo
                                    print(f"Ruido de fondo - Energy: {energy:.6f}")
                                        
                    except queue.Empty:
                        continue
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        continue
                        
        except KeyboardInterrupt:
            print("\nSistema detenido")
        except Exception as e:
            print(f"Error en sistema: {e}")

def main():
    """Función principal"""
    sistema = SimpleWakeWordSystem()
    sistema.ejecutar_sistema()

if __name__ == "__main__":
    main()
