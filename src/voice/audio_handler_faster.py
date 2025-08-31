import logging
from typing import Optional, Callable
import queue
import numpy as np
import pyaudio
import uuid
import wave
import os
import time
import faster_whisper
from datetime import datetime
from pathlib import Path
from scipy.signal import find_peaks
from ..config.config import setup_logging, load_config

class AudioHandlerFaster:
    def __init__(self, config_path: str = "config/settings.json"):
        """Inicializar AudioHandler con Faster-Whisper"""
        try:
            # Configurar logging
            setup_logging()
            self.logger = logging.getLogger("AudioHandlerFaster")
            
            # Cargar configuración
            self.config = load_config(config_path)
            
            # Estado
            self.is_listening = False
            self.is_recording = False
            self.recording_start_time = None
            self.chunk_buffer = []
            
            # Buffers
            self.audio_buffer = queue.Queue()
            
            # Callbacks
            self.on_speech_detected: Optional[Callable[[bytes], None]] = None
            self.on_silence_detected: Optional[Callable[[], None]] = None
            self.on_transcription: Optional[Callable[[str], None]] = None
            
            # PyAudio
            self.audio = None
            self.stream = None
            
            # Configuración de audio
            self.sample_rate = self.config['audio']['sample_rate']
            self.chunk_size = self.config['audio']['chunk_size']
            self.buffer_seconds = self.config['audio']['buffer_seconds']
            self.silence_threshold = self.config['audio']['silence_threshold']
            self.input_device = self.config['audio']['input_device']
            
            # Mostrar dispositivos de audio disponibles
            self.logger.info("Dispositivos de audio disponibles:")
            for i in range(pyaudio.PyAudio().get_device_count()):
                device_info = pyaudio.PyAudio().get_device_info_by_index(i)
                try:
                    name = device_info['name'].encode('latin1').decode('utf-8')
                except UnicodeDecodeError:
                    name = device_info['name']
                self.logger.info(f"Dispositivo {i}: {name}")
            
            # Inicializar Faster-Whisper
            self.logger.info("Cargando modelo Faster-Whisper...")
            model_name = self.config['audio'].get('whisper_model', 'small')
            self.model = faster_whisper.WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8"
            )
            self.logger.info("Modelo Faster-Whisper cargado exitosamente")
            
            # Estado de la aplicación
            self.last_state_change = time.time()
            self.MAX_IDLE_TIME = 60  # 1 minuto en segundos
            self.state = "INACTIVE"
            
        except Exception as e:
            self.logger.error(f"Error al inicializar AudioHandlerFaster: {e}")
            raise

    def start(self):
        """Iniciar la detección de voz"""
        try:
            self.logger.info("Iniciando detector de voz...")
            self.is_listening = True
            
            # Inicializar PyAudio
            self.audio = pyaudio.PyAudio()
            
            # Configurar stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.input_device,
                stream_callback=self._audio_callback,
                start=True
            )
            
            self.stream.start_stream()
            self.logger.info("Stream iniciado exitosamente")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error al configurar el stream: {e}")
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.audio:
                self.audio.terminate()
            raise
            
    def stop(self):
        """Detener la detección de voz"""
        try:
            self.logger.info("Deteniendo detector")
            self.is_listening = False
            
            # Detener y cerrar el stream
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
                except Exception as e:
                    self.logger.error(f"Error al cerrar el stream: {e}")
            
            # Terminar PyAudio
            if self.audio:
                try:
                    self.audio.terminate()
                    self.audio = None
                except Exception as e:
                    self.logger.error(f"Error al terminar PyAudio: {e}")
            
            self.logger.info("Detector detenido exitosamente")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al detener el detector: {e}")
            return False

    def set_on_transcription_callback(self, callback: Callable[[str], None]):
        """Establecer callback para cuando se complete una transcripción"""
        self.on_transcription = callback

    def _transcribe_audio(self, audio_path: str):
        """Transcribir el archivo de audio usando Faster-Whisper"""
        try:
            if not hasattr(self, 'model'):
                self.logger.warning("Modelo Faster-Whisper no cargado, omitiendo transcripción")
                return
                
            self.logger.info("Iniciando transcripción con Faster-Whisper...")
            start_time = time.time()
            
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=5,
                language='es',
                vad_filter=True
            )
            
            # Recopilar transcripción
            transcription = ""
            for segment in segments:
                transcription += segment.text + " "
            
            transcription = transcription.strip()
            transcribe_time = time.time() - start_time
            
            if transcription:
                self.logger.info(f"Transcripción completada en {transcribe_time:.2f}s: {transcription}")
                # Llamar al callback de transcripción si está definido
                if self.on_transcription:
                    self.on_transcription(transcription)
            else:
                self.logger.warning("No se detectó habla en el audio")
                
        except Exception as e:
            self.logger.error(f"Error en la transcripción: {e}")
        finally:
            # Limpiar el archivo temporal
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                self.logger.warning(f"No se pudo eliminar el archivo temporal: {e}")
                
    def _has_audio_peaks(self, audio_chunk: bytes, peak_threshold=1000, min_peaks=1):
        """Detecta si hay picos de audio significativos en el chunk"""
        try:
            if not audio_chunk or len(audio_chunk) < 2:
                return False
            
            # Convertir a numpy array y obtener valores absolutos
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            abs_audio = np.abs(audio_data)
            
            # Encontrar picos usando el umbral especificado
            peaks, props = find_peaks(abs_audio, height=peak_threshold)
            
            # Logging para depuración
            self.logger.debug(f"Picos detectados: {len(peaks)}, Alturas: {props.get('peak_heights', [])}")
            
            # Retornar True si hay suficientes picos
            return len(peaks) >= min_peaks
            
        except Exception as e:
            self.logger.error(f"Error en _has_audio_peaks: {e}")
            return False

    def _is_speech(self, audio_chunk: bytes) -> bool:
        """Detectar si hay voz en el chunk de audio"""
        try:
            if not audio_chunk or len(audio_chunk) < 2:
                self.logger.debug("Chunk de audio vacío o demasiado pequeño")
                return False
            
            # Convertir a numpy array
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Validar que el array tenga valores válidos
            if np.isnan(audio_np).any() or np.isinf(audio_np).any():
                self.logger.debug("Valores NaN o Inf encontrados en el audio")
                return False
            
            # Calcular el RMS (Root Mean Square)
            abs_audio = np.fabs(audio_np)
            rms = np.sqrt(np.mean(np.square(abs_audio)))
            
            # Logging detallado de los valores
            self.logger.debug(f"RMS: {rms:.2f}, Umbral: {self.silence_threshold}")
            
            # Si el RMS supera el umbral de silencio, se considera voz
            is_speech = rms > self.silence_threshold
            if is_speech:
                self.logger.info(f"Voz detectada! RMS: {rms:.2f}, Umbral: {self.silence_threshold}")
                # Llamar al callback solo si está definido y puede recibir el argumento
                if self.on_speech_detected:
                    try:
                        self.on_speech_detected(audio_chunk)
                    except TypeError:
                        # Si el callback no acepta argumentos, llamarlo sin argumentos
                        self.on_speech_detected()
            else:
                self.logger.debug(f"Silencio detectado. RMS: {rms:.2f}")
                # Llamar al callback de silencio si está definido
                if self.on_silence_detected:
                    self.on_silence_detected()
            
            return is_speech
            
        except Exception as e:
            self.logger.error(f"Error en _is_speech: {e}")
            return False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback para el stream de PyAudio"""
        try:
            if not self.is_listening:
                return (in_data, pyaudio.paComplete)
            
            # Inicializar el buffer si no existe
            if not hasattr(self, 'chunk_buffer'):
                self.chunk_buffer = []
            
            # Mantener un buffer circular de los últimos 10 segundos
            self.chunk_buffer.append(in_data)
            max_buffer_size = self.sample_rate * 10 // self.chunk_size
            if len(self.chunk_buffer) > max_buffer_size:
                self.chunk_buffer = self.chunk_buffer[-max_buffer_size:]
            
            # Verificar si hay voz en el chunk actual
            if self._is_speech(in_data):
                if not self.is_recording:
                    self.logger.info("Voz detectada, iniciando grabación...")
                    self.is_recording = True
                    self.recording_start_time = time.time()
            
            # Si estamos grabando, verificar si es hora de guardar
            if self.is_recording:
                current_time = time.time()
                if current_time - self.recording_start_time >= self.buffer_seconds:
                    self.logger.info("Tiempo de grabación alcanzado, guardando...")
                    try:
                        self._save_recording()
                    except Exception as save_error:
                        self.logger.error(f"Error al guardar grabación: {save_error}")
                    finally:
                        # Mantener el buffer pero marcar como no grabando
                        self.is_recording = False
                
            return (in_data, pyaudio.paContinue)
            
        except Exception as e:
            self.logger.error(f"Error en _audio_callback: {e}")
            if self.is_recording:
                self.is_recording = False
                self.recording_start_time = None
            return (in_data, pyaudio.paContinue)

    def _save_recording(self):
        """Guardar la grabación actual y transcribirla"""
        try:
            if not hasattr(self, 'chunk_buffer') or not self.chunk_buffer:
                self.logger.warning("Buffer vacío, omitiendo guardado")
                return
            
            # Crear directorio temporal si no existe
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Crear nombre de archivo único
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            recording_id = str(uuid.uuid4())[:8]
            temp_path = os.path.join(temp_dir, f'recording_{timestamp}_{recording_id}.wav')
            
            # Guardar audio en WAV
            wf = wave.open(temp_path, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.chunk_buffer))
            wf.close()
            
            self.logger.info(f"Grabación guardada en: {temp_path}")
            
            # Transcribir con Faster-Whisper
            self._transcribe_audio(temp_path)
                
        except Exception as e:
            self.logger.error(f"Error al guardar la grabación: {e}")
            self.is_recording = False
            self.recording_start_time = None 