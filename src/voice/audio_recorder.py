import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import logging
import os
from datetime import datetime
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AudioRecorder")

class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, output_dir: str = "data/recordings"):
        """
        Inicializa el grabador de audio
        
        Args:
            sample_rate: Tasa de muestreo en Hz
            output_dir: Directorio donde se guardarán las grabaciones
        """
        self.sample_rate = sample_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Estado
        self.is_recording = False
        self.recording = None
    
    def record_audio(self, duration: float = 5.0) -> str:
        """
        Graba audio durante la duración especificada
        
        Args:
            duration: Duración de la grabación en segundos
            
        Returns:
            str: Ruta al archivo de audio grabado
        """
        try:
            logger.info(f"Iniciando grabación de {duration} segundos...")
            self.is_recording = True
            
            # Grabar audio
            self.recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            
            # Esperar a que termine la grabación
            sd.wait()
            
            # Generar nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"recording_{timestamp}.wav"
            
            # Guardar como WAV
            wav.write(output_file, self.sample_rate, self.recording)
            
            logger.info(f"Grabación guardada en: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error al grabar audio: {e}")
            raise
        finally:
            self.is_recording = False

def record_audio_cli():
    """Interfaz de línea de comandos para grabar audio"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Grabador de audio')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Duración de la grabación en segundos')
    parser.add_argument('--output-dir', type=str, default="data/recordings",
                       help='Directorio de salida para las grabaciones')
    
    args = parser.parse_args()
    
    try:
        recorder = AudioRecorder(output_dir=args.output_dir)
        output_file = recorder.record_audio(args.duration)
        print(f"\n¡Grabación completada! Archivo guardado en: {output_file}")
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    record_audio_cli()
