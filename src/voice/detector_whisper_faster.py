import logging
from typing import Optional, Callable
import sys
import os
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd

# Agregar el directorio raíz del proyecto a PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.config import setup_logging
from src.voice.audio_handler_faster import AudioHandlerFaster

class VoiceDetectorFaster:
    def __init__(self, config_path: str = "config/settings.json"):
        """Inicializar el detector de voz con Faster-Whisper"""
        try:
            # Configurar logging
            setup_logging()
            self.logger = logging.getLogger("VoiceDetectorFaster")
            self.logger.info("Inicializando VoiceDetectorFaster")
            
            # Inicializar AudioHandlerFaster
            self.audio_handler = AudioHandlerFaster(config_path)
            
            # Configurar callbacks
            self.audio_handler.on_speech_detected = self._on_speech_detected
            self.audio_handler.on_silence_detected = self._on_silence_detected
            self.audio_handler.on_transcription = self._on_transcription_complete
            
            # Callbacks de usuario
            self.on_speech: Optional[Callable[[], None]] = None
            self.on_silence: Optional[Callable[[], None]] = None
            self.on_text: Optional[Callable[[str], None]] = None
            
        except Exception as e:
            self.logger.error(f"Error al inicializar VoiceDetectorFaster: {e}")
            raise

    def start(self):
        """Iniciar el detector"""
        try:
            self.logger.info("Iniciando detector de voz con Faster-Whisper...")
            return self.audio_handler.start()
        except Exception as e:
            self.logger.error(f"Error al iniciar el detector: {e}")
            return False

    def stop(self):
        """Detener el detector"""
        try:
            self.logger.info("Deteniendo detector de voz...")
            return self.audio_handler.stop()
        except Exception as e:
            self.logger.error(f"Error al detener el detector: {e}")
            return False

    def set_speech_callback(self, callback: Callable[[], None]):
        """Establecer callback para cuando se detecta voz"""
        self.on_speech = callback
        self.logger.info("Callback de voz configurado")

    def set_silence_callback(self, callback: Callable[[], None]):
        """Establecer callback para cuando se detecta silencio"""
        self.on_silence = callback
        self.logger.info("Callback de silencio configurado")

    def set_text_callback(self, callback: Callable[[str], None]):
        """Establecer callback para cuando se completa la transcripción"""
        self.on_text = callback
        self.logger.info("Callback de texto configurado")

    def _on_speech_detected(self, audio_chunk: bytes = None):
        """Callback interno para cuando se detecta voz"""
        if self.on_speech:
            try:
                self.on_speech()
                self.logger.debug("Callback de voz ejecutado correctamente")
            except Exception as e:
                self.logger.error(f"Error en callback de voz: {e}")

    def _on_silence_detected(self):
        """Callback interno para cuando se detecta silencio"""
        if self.on_silence:
            try:
                self.on_silence()
                self.logger.debug("Callback de silencio ejecutado correctamente")
            except Exception as e:
                self.logger.error(f"Error en callback de silencio: {e}")

    def _on_transcription_complete(self, text: str):
        """Callback interno para cuando se completa la transcripción"""
        if self.on_text:
            try:
                self.on_text(text)
                self.logger.info(f"Transcripción procesada: {text}")
            except Exception as e:
                self.logger.error(f"Error en callback de texto: {e}")

def create_voice_detector(config_path: str = "config/settings.json") -> VoiceDetectorFaster:
    """
    Crea una instancia del detector de voz con Faster-Whisper
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        VoiceDetectorFaster: Instancia del detector
    """
    return VoiceDetectorFaster(config_path)

if __name__ == "__main__":
    # Ejemplo de uso
    def on_speech_detected():
        print("🎤 ¡Voz detectada!")
    
    def on_silence_detected():
        print("🔇 Silencio detectado")
    
    def on_text_transcribed(text: str):
        print(f"📝 Transcripción: {text}")
    
    try:
        # Crear detector
        detector = create_voice_detector()
        
        # Configurar callbacks
        detector.set_speech_callback(on_speech_detected)
        detector.set_silence_callback(on_silence_detected)
        detector.set_text_callback(on_text_transcribed)
        
        print("🎧 Iniciando detector de voz...")
        print("💬 Habla para probar el sistema")
        print("⏹️  Presiona Ctrl+C para salir")
        
        # Iniciar detector
        detector.start()
        
        # Mantener ejecutando
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo detector...")
        detector.stop()
        print("✅ Detector detenido correctamente")
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'detector' in locals():
            detector.stop() 