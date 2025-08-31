import logging
import os
import time
from typing import Optional, Dict, Any, List
import faster_whisper

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TranscriptionServiceFaster")

class TranscriptionServiceFaster:
    def __init__(self, model_name: str = 'small', device: str = None, compute_type: str = 'int8'):
        """Inicializa el servicio de transcripción con Faster-Whisper
        
        Args:
            model_name: Nombre del modelo (tiny, base, small, medium, large, large-v2)
            device: Dispositivo a usar (cuda, cpu, mps). Si es None, se detecta automáticamente
            compute_type: Tipo de computación (int8, float16, float32)
        """
        self.model_name = model_name
        self.device = device or 'cpu'  # Faster-Whisper funciona mejor en CPU por defecto
        self.compute_type = compute_type
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Carga el modelo de Faster-Whisper"""
        try:
            logger.info(f"Cargando modelo Faster-Whisper: {self.model_name} en {self.device}")
            start_time = time.time()
            
            # Cargar modelo de transcripción
            self.model = faster_whisper.WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
            
            load_time = time.time() - start_time
            logger.info(f"Modelo cargado exitosamente en {load_time:.2f} segundos")
            
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str, beam_size: int = 5, language: str = None) -> Optional[Dict[str, Any]]:
        """
        Transcribe un archivo de audio usando Faster-Whisper
        
        Args:
            audio_path: Ruta al archivo de audio
            beam_size: Tamaño del beam search (mayor = más preciso, más lento)
            language: Idioma del audio (None para detección automática)
            
        Returns:
            dict: Resultado de la transcripción con segmentos y texto completo
                 o None si hay un error
        """
        if not os.path.exists(audio_path):
            logger.error(f"El archivo {audio_path} no existe")
            return None
            
        try:
            logger.info(f"Transcribiendo archivo: {audio_path}")
            start_time = time.time()
            
            # Transcribir el audio
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=beam_size,
                language=language,
                vad_filter=True,  # Filtrar silencios
                vad_parameters=dict(min_silence_duration_ms=500)  # Silencios de al menos 500ms
            )
            
            # Recopilar resultados
            transcription = ""
            segments_list = []
            
            for segment in segments:
                transcription += segment.text + " "
                segments_list.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'words': getattr(segment, 'words', [])  # Palabras con timestamps si están disponibles
                })
            
            transcribe_time = time.time() - start_time
            
            if transcription.strip():
                logger.info(f"Transcripción completada en {transcribe_time:.2f} segundos")
                return {
                    "text": transcription.strip(),
                    "segments": segments_list,
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "duration": info.duration,
                    "processing_time": transcribe_time
                }
            else:
                logger.warning("No se detectó habla en el audio")
                return None
                
        except Exception as e:
            logger.error(f"Error al transcribir con Faster-Whisper: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def transcribe_with_timestamps(self, audio_path: str, word_timestamps: bool = True) -> Optional[Dict[str, Any]]:
        """
        Transcribe con timestamps de palabras (si está disponible)
        
        Args:
            audio_path: Ruta al archivo de audio
            word_timestamps: Si incluir timestamps de palabras individuales
            
        Returns:
            dict: Resultado con timestamps detallados
        """
        if not os.path.exists(audio_path):
            logger.error(f"El archivo {audio_path} no existe")
            return None
            
        try:
            logger.info(f"Transcribiendo con timestamps: {audio_path}")
            
            # Transcribir con timestamps de palabras
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=5,
                word_timestamps=word_timestamps,
                vad_filter=True
            )
            
            # Procesar resultados con timestamps
            transcription = ""
            segments_list = []
            
            for segment in segments:
                segment_data = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip()
                }
                
                if hasattr(segment, 'words') and segment.words:
                    segment_data['words'] = [
                        {
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'probability': word.probability
                        }
                        for word in segment.words
                    ]
                
                segments_list.append(segment_data)
                transcription += segment.text + " "
            
            if transcription.strip():
                return {
                    "text": transcription.strip(),
                    "segments": segments_list,
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "duration": info.duration
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error al transcribir con timestamps: {e}")
            return None

def create_transcription_service(config_path: str = "config/settings.json") -> TranscriptionServiceFaster:
    """
    Crea una instancia del servicio de transcripción con Faster-Whisper
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        TranscriptionServiceFaster: Instancia del servicio
    """
    # Por ahora usamos el modelo por defecto, pero podríamos cargarlo desde la configuración
    return TranscriptionServiceFaster()

if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python -m src.voice.transcription_service_faster <ruta_al_audio>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    service = create_transcription_service()
    transcription = service.transcribe_audio(audio_file)
    
    if transcription:
        print(f"\nTranscripción: {transcription['text']}")
        print(f"Idioma: {transcription['language']} (probabilidad: {transcription['language_probability']:.2f})")
        print(f"Duración: {transcription['duration']:.2f} segundos")
        print(f"Tiempo de procesamiento: {transcription['processing_time']:.2f} segundos")
    else:
        print("No se pudo realizar la transcripción") 