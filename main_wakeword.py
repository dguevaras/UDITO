
#!/usr/bin/env python3
"""
UDI Sistema de Wake Word - Interfaz Principal de Pruebas
Interfaz profesional de desarrollador para pruebas y estadísticas del wake word
"""

# Configurar TensorFlow para evitar colgadas
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Solo errores críticos
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Deshabilitar optimizaciones problemáticas

import time
import queue
import numpy as np
import sounddevice as sd
import tensorflow as tf
from collections import deque
from datetime import datetime

# Configuración
CONFIG = {
    "model_path": "WakeWord-project/wakeword_model.tflite",
    "sample_rate": 16000,
    "block_duration": 1.0,
    "threshold_voice": 0.0002,
    "wakeword_threshold": 0.49,
    "chunk_ms": 80,
    "sensitivity": 0.6,
    "calibrate_secs": 3.0,
    "cool_down_sec": 0.5  # Reducido de 1.0 a 0.5 segundos
}

class ProbadorWakeWord:
    def __init__(self):
        self.estadisticas = {
            "activaciones": 0,
            "falsos_positivos": 0,
            "total_muestras": 0,
            "tiempo_inicio": time.time(),
            "ultima_activacion": None,
            "tiempo_respuesta_promedio": 0.0,
            "tiempos_respuesta": [],
            "ultima_deteccion": None,
            "resultado_ultima": None,
            "probabilidad_ultima": None,
            "detectado_como_wakeword": False
        }
        
        self.estado_sistema = "INICIANDO"
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
    def mostrar_encabezado(self):
        """Mostrar encabezado profesional"""
        print("=" * 80)
        print("                                SISTEMA UDI WAKE WORD")
        print("=" * 80)
        print(f"MODELO: {CONFIG['model_path']}")
        print(f"PALABRA CLAVE: UDITO")
        print(f"FRECUENCIA MUESTREO: {CONFIG['sample_rate']} Hz")
        print(f"UMBRAL: {CONFIG['wakeword_threshold']}")
        print(f"ESTADO: {self.estado_sistema}")
        print("=" * 80)
        
    def mostrar_estadisticas(self):
        """Mostrar estadísticas actuales"""
        tiempo_activo = time.time() - self.estadisticas["tiempo_inicio"]
        print("\n" + "=" * 80)
        print("ESTADÍSTICAS                    RENDIMIENTO")
        print("=" * 80)
        print(f"Activaciones: {self.estadisticas['activaciones']:<15} Tiempo Activo: {tiempo_activo:.1f}s")
        print(f"Falsos Positivos: {self.estadisticas['falsos_positivos']:<10} Respuesta Prom: {self.estadisticas['tiempo_respuesta_promedio']:.0f}ms")
        if self.estadisticas['ultima_activacion']:
            tiempo_desde = time.time() - self.estadisticas['ultima_activacion']
            print(f"Última Activación: {tiempo_desde:.1f}s atrás")
        print(f"Total Muestras: {self.estadisticas['total_muestras']}")
        
        # Mostrar información de la última detección
        if self.estadisticas['ultima_deteccion']:
            tiempo_desde_det = time.time() - self.estadisticas['ultima_deteccion']
            print(f"\nÚLTIMA DETECCIÓN:")
            print(f"  Resultado: {self.estadisticas['resultado_ultima']}")
            print(f"  Probabilidad: {self.estadisticas['probabilidad_ultima']:.3f}")
            print(f"  Hace: {tiempo_desde_det:.1f}s")
            print(f"  Detectado como Wakeword: {'SÍ' if self.estadisticas['detectado_como_wakeword'] else 'NO'}")
        
        print("=" * 80)
        
    def cargar_modelo(self):
        """Cargar modelo TensorFlow Lite con timeout"""
        print("[INFO] Cargando modelo TensorFlow Lite...")
        try:
            import threading
            import time
            
            resultado = {"exito": False, "error": None}
            
            def cargar_modelo_thread():
                try:
                    self.interpreter = tf.lite.Interpreter(model_path=CONFIG['model_path'])
                    self.interpreter.allocate_tensors()
                    self.input_details = self.interpreter.get_input_details()
                    self.output_details = self.interpreter.get_output_details()
                    resultado["exito"] = True
                except Exception as e:
                    resultado["error"] = str(e)
            
            # Ejecutar en thread separado con timeout
            thread = threading.Thread(target=cargar_modelo_thread)
            thread.daemon = True
            thread.start()
            
            # Esperar máximo 30 segundos
            thread.join(timeout=30)
            
            if thread.is_alive():
                print("[ERROR] Timeout: El modelo tardó demasiado en cargar")
                return False
            
            if resultado["exito"]:
                print(f"[INFO] Modelo cargado exitosamente")
                print(f"[INFO] Forma de entrada: {self.input_details[0]['shape']}")
                print(f"[INFO] Forma de salida: {self.output_details[0]['shape']}")
                return True
            else:
                print(f"[ERROR] Error al cargar modelo: {resultado['error']}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Falló al cargar modelo: {e}")
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
        except Exception as e:
            print(f"[ERROR] Falló extracción MFCC: {e}")
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
        except Exception as e:
            print(f"[ERROR] Falló inferencia: {e}")
            return 0.5
            
    def on_wake_word_detectado(self, confianza, tiempo_respuesta):
        """Manejar detección de wake word"""
        self.estadisticas["activaciones"] += 1
        self.estadisticas["ultima_activacion"] = time.time()
        self.estadisticas["tiempos_respuesta"].append(tiempo_respuesta)
        self.estadisticas["tiempo_respuesta_promedio"] = np.mean(self.estadisticas["tiempos_respuesta"])
        
        print("\n" + "=" * 80)
        print("🎯 ¡WAKE WORD ACTIVADO!")
        print("=" * 80)
        print(f"CONFIANZA: {confianza:.1%}")
        print(f"TIEMPO RESPUESTA: {tiempo_respuesta:.0f}ms")
        print(f"ACTIVACIÓN #{self.estadisticas['activaciones']}")
        print(f"MARCA TIEMPO: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print("[ESTADO] Sistema activado - Wake word deshabilitado")
        print("[INFO] Listo para comandos de voz...")
        print("=" * 80)
        
    def confirmar_deteccion(self, confianza):
        """Preguntar al usuario si fue realmente UDITO"""
        print(f"\n[CONFIRMACIÓN] Se detectó con {confianza:.1%} de confianza")
        print("¿Era realmente 'UDITO'?")
        respuesta = input("Respuesta (s/n): ").lower().strip()
        
        if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
            self.estadisticas['activaciones'] += 1
            self.estadisticas['resultado_ultima'] = "WAKEWORD CONFIRMADO"
            self.estadisticas['detectado_como_wakeword'] = True
            print("[INFO] ✅ Wakeword confirmado - Activando sistema")
            return True
        else:
            self.estadisticas['falsos_positivos'] += 1
            self.estadisticas['resultado_ultima'] = "FALSO POSITIVO"
            self.estadisticas['detectado_como_wakeword'] = False
            print("[INFO] ❌ Falso positivo registrado")
            return False
            
    def ejecutar_prueba(self):
        """Bucle principal de pruebas - VERSIÓN SIMPLIFICADA"""
        if not self.cargar_modelo():
            return
            
        print("[INFO] Iniciando stream de audio...")
        self.estado_sistema = "LISTO"
        self.mostrar_encabezado()
        
        # Pausa para que el usuario esté listo
        print("\n" + "=" * 80)
        print("                                PREPARACIÓN")
        print("=" * 80)
        print("[INFO] Sistema configurado y listo")
        print("[INFO] Asegúrate de estar en un ambiente tranquilo")
        print("[INFO] El micrófono comenzará a escuchar cuando presiones Enter")
        print("=" * 80)
        input("[ENTRADA] Presiona Enter para iniciar la detección...")
        print("[INFO] Iniciando detección de wake word...")
        print("=" * 80)
        
        audio_q = queue.Queue()
        block_samples = int(CONFIG['sample_rate'] * CONFIG['block_duration'])
        
        def callback(indata, frames, ctime, status):
            if status:
                print(f"[ADVERTENCIA] Estado audio: {status}")
            audio_q.put(indata.copy().reshape(-1))
            
        try:
            print("[INFO] Configurando dispositivo de audio...")
            with sd.InputStream(
                samplerate=CONFIG['sample_rate'],
                channels=1,
                dtype=np.float32,
                blocksize=int(CONFIG['sample_rate'] * 0.1),
                callback=callback
            ):
                print("[INFO] Sistema listo - Escuchando wake word")
                print("[INFO] Di 'UDITO' para activar el sistema")
                print("=" * 80)
                
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
                                start_time = time.time()
                                
                                # Mostrar que se detectó audio
                                print("AUDIO: Voz detectada - Procesando...")
                                
                                # Ejecutar inferencia
                                prob = self.ejecutar_inferencia(block)
                                tiempo_respuesta = (time.time() - start_time) * 1000
                                
                                probs_hist.append(prob)
                                avg_prob = float(np.mean(probs_hist))
                                
                                self.estadisticas["total_muestras"] += 1
                                
                                # Mostrar resultado del análisis
                                if avg_prob >= CONFIG['wakeword_threshold']:
                                    consecutive_hits += 1
                                    if consecutive_hits >= 2:  # Solo mostrar cuando se confirma
                                        current_time = time.time()
                                        if current_time - last_detection > CONFIG['cool_down_sec']:
                                            print(f"\nDETECCION: Wake Word detectado (Probabilidad: {avg_prob:.1%})")
                                            print("Confirmacion: Era realmente 'UDITO'? (s/n): ", end="")
                                            respuesta = input().lower().strip()
                                            
                                            if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
                                                self.estadisticas['activaciones'] += 1
                                                self.estadisticas['resultado_ultima'] = "WAKEWORD CONFIRMADO"
                                                self.estadisticas['detectado_como_wakeword'] = True
                                                print("RESULTADO: Confirmado - Sistema activado")
                                            else:
                                                self.estadisticas['falsos_positivos'] += 1
                                                self.estadisticas['resultado_ultima'] = "FALSO POSITIVO"
                                                self.estadisticas['detectado_como_wakeword'] = False
                                                print("RESULTADO: Falso positivo registrado")
                                            
                                            # Actualizar estadísticas
                                            self.estadisticas['ultima_deteccion'] = time.time()
                                            self.estadisticas['probabilidad_ultima'] = avg_prob
                                            
                                            last_detection = current_time
                                            consecutive_hits = 0
                                            probs_hist.clear()
                                            
                                            print("ESTADO: Continuando escucha...")
                                            print("-" * 50)
                                    else:
                                        print(f"WAKE: Hit {consecutive_hits}/2 (Prob: {avg_prob:.1%})")
                                else:
                                    consecutive_hits = 0
                                    print(f"RESULTADO: No es wakeword (Prob: {avg_prob:.1%})")
                            else:
                                # Mostrar que está escuchando (cada 5 segundos)
                                current_time = time.time()
                                if not hasattr(self, 'last_listening_msg') or current_time - getattr(self, 'last_listening_msg', 0) > 5:
                                    print("ESTADO: Escuchando...")
                                    self.last_listening_msg = current_time
                                        
                    except queue.Empty:
                        continue
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        print(f"[ERROR] Error de procesamiento: {e}")
                        continue
                        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupción recibida - Cerrando...")
        except Exception as e:
            print(f"[ERROR] Error en stream de audio: {e}")
        finally:
            self.mostrar_estadisticas_finales()
            
    def mostrar_estadisticas_finales(self):
        """Mostrar estadísticas finales"""
        print("\n" + "=" * 80)
        print("                                ESTADÍSTICAS FINALES")
        print("=" * 80)
        tiempo_activo = time.time() - self.estadisticas["tiempo_inicio"]
        print(f"Tiempo Total Ejecución: {tiempo_activo:.1f} segundos")
        print(f"Activaciones Wake Word: {self.estadisticas['activaciones']}")
        print(f"Falsos Positivos: {self.estadisticas['falsos_positivos']}")
        print(f"Total Muestras Audio: {self.estadisticas['total_muestras']}")
        if self.estadisticas['tiempos_respuesta']:
            print(f"Tiempo Respuesta Promedio: {np.mean(self.estadisticas['tiempos_respuesta']):.0f}ms")
            print(f"Mejor Tiempo Respuesta: {min(self.estadisticas['tiempos_respuesta']):.0f}ms")
        print("=" * 80)

def main():
    """Punto de entrada principal"""
    print("[INFO] Iniciando Probador de Wake Word...")
    probador = ProbadorWakeWord()
    probador.ejecutar_prueba()

if __name__ == "__main__":
    main()
