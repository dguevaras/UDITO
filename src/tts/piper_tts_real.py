#!/usr/bin/env python3
"""
Módulo TTS usando Piper TTS real con voces naturales
"""

import json
import subprocess
import tempfile
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Configurar logging
logger = logging.getLogger(__name__)

class PiperTTS:
    """Sintetizador de voz usando Piper TTS real"""
    
    def __init__(self, config_path: str = "config/tts_config.json"):
        """Inicializa el TTS de Piper"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Ruta al ejecutable de Piper
        self.piper_path = Path("piper/piper.exe")
        
        # Ruta a las voces (dentro de la carpeta piper)
        self.voices_path = Path("piper/voices")
        
        # Verificar que Piper esté disponible
        if not self.piper_path.exists():
            raise FileNotFoundError(f"Piper TTS no encontrado en: {self.piper_path}")
        
        # Inicializar voz
        self._initialize_voice()
        
        logger.info("Piper TTS inicializado correctamente")
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga configuración del TTS"""
        default_config = {
            "voice_rate": 1.0,
            "voice_volume": 1.0,
            "voice_model": "es_ES-sharvard-medium.onnx",
            "assistant_name": "UDI",
            "greeting": "Hola, soy UDI, tu asistente universitario. ¿En qué puedo ayudarte?",
            "farewell": "Gracias por usar UDI. ¡Que tengas un buen día!",
            "thinking": "Déjame buscar esa información...",
            "not_found": "Lo siento, no encontré información sobre eso.",
            "error": "Tuve un problema al procesar tu consulta."
        }
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Actualizar con valores por defecto si faltan
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    logger.info(f"Configuración TTS cargada desde {self.config_path}")
                    return config
            else:
                logger.info("Archivo de configuración no encontrado, usando valores por defecto")
                return default_config
                
        except Exception as e:
            logger.error(f"Error al cargar configuración: {e}")
            return default_config
    
    def _save_config(self):
        """Guarda la configuración actual"""
        try:
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info("Configuración TTS guardada")
        except Exception as e:
            logger.error(f"Error al guardar configuración: {e}")
    
    def _initialize_voice(self):
        """Inicializa la voz de Piper"""
        try:
            # Buscar el modelo de voz
            voice_model = self.config.get('voice_model', 'es_MX-ald-medium.onnx')
            voice_path = self.voices_path / voice_model
            
            if not voice_path.exists():
                logger.warning(f"Modelo de voz no encontrado: {voice_path}")
                logger.info("Usando voz por defecto del sistema")
                return
            
            # Verificar archivo de configuración
            config_path = voice_path.with_suffix('.onnx.json')
            if not config_path.exists():
                logger.warning(f"Archivo de configuración de voz no encontrado: {config_path}")
            
            logger.info(f"Voz configurada: {voice_model}")
            
        except Exception as e:
            logger.error(f"Error al inicializar voz: {e}")
    
    def speak(self, text: str, wait: bool = True):
        """Reproduce texto usando Piper TTS"""
        try:
            if not text.strip():
                return
            
            # Ajustar texto para pronunciación natural
            text = self._adjust_text_for_pronunciation(text)
            
            logger.info(f"Reproduciendo con Piper: {text[:50]}...")
            
            # Crear archivos temporales
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w', encoding='utf-8') as temp_text:
                temp_text.write(text)
                temp_text_path = temp_text.name
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # Comando para Piper
            voice_model = self.config.get('voice_model', 'es_MX-ald-medium.onnx')
            voice_path = self.voices_path / voice_model
            
            if voice_path.exists():
                # Usar modelo específico
                cmd = [
                    str(self.piper_path),
                    '--model', str(voice_path),
                    '--output_file', temp_audio_path
                ]
            else:
                # Usar voz por defecto
                cmd = [
                    str(self.piper_path),
                    '--output_file', temp_audio_path
                ]
            
            # Ejecutar Piper con archivo de entrada
            with open(temp_text_path, 'r', encoding='utf-8') as input_file:
                process = subprocess.Popen(
                    cmd,
                    stdin=input_file,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                stdout, stderr = process.communicate()
            
            # Limpiar archivo de texto temporal
            try:
                os.unlink(temp_text_path)
            except:
                pass
            
            if process.returncode != 0:
                logger.error(f"Error en Piper TTS: {stderr}")
                return
            
            # Verificar que se generó el archivo de audio
            if not os.path.exists(temp_audio_path):
                logger.error("No se generó archivo de audio")
                return
            
            # Reproducir audio
            self._play_audio(temp_audio_path)
            
            # Limpiar archivo de audio temporal
            try:
                os.unlink(temp_audio_path)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error al reproducir voz: {e}")
    
    def _adjust_text_for_pronunciation(self, text: str) -> str:
        """Ajusta el texto para pronunciación natural"""
        # Forzar pronunciación natural del nombre
        name_pron = self.config.get('assistant_name', 'Udi')
        if name_pron.strip().lower() == 'udi':
            text = text.replace('UDI', 'Udi').replace('U.D.I.', 'Udi').replace('U. D. I.', 'Udi')
        elif name_pron.strip().lower() == 'udito':
            text = text.replace('UDITO', 'Udito')
        
        return text
    
    def _play_audio(self, audio_path: str):
        """Reproduce el archivo de audio"""
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # Esperar a que termine
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        except ImportError:
            # Fallback: usar playsound si pygame no está disponible
            try:
                from playsound import playsound
                playsound(audio_path)
            except ImportError:
                logger.warning("No se pudo reproducir audio. Instala pygame o playsound")
        except Exception as e:
            logger.error(f"Error al reproducir audio: {e}")
    
    def speak_response(self, response_type: str, wait: bool = True):
        """Reproduce una respuesta pre-entrenada"""
        try:
            responses = self._get_responses()
            if response_type in responses:
                import random
                text = random.choice(responses[response_type])
                self.speak(text, wait)
                return text
            else:
                logger.warning(f"Tipo de respuesta no encontrado: {response_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error al reproducir respuesta: {e}")
            return None
    
    def _get_responses(self) -> Dict[str, list]:
        """Obtiene respuestas pre-entrenadas"""
        assistant_name = self.config.get('assistant_name', 'UDI')
        # Presentación extendida siempre
        presentacion = f"Hola, soy {assistant_name} de la universidad UDIT (Universidad de Diseño, Innovación y Tecnología). ¿En qué puedo ayudarte?"
        
        return {
            "greeting": [presentacion],
            "farewell": [
                f"Gracias por usar {assistant_name}. ¡Que tengas un buen día!",
                f"Ha sido un placer ayudarte. ¡Hasta luego!",
                f"¡Que tengas éxito en tus estudios! Hasta la próxima."
            ],
            "thinking": [
                "Déjame buscar esa información...",
                "Un momento, estoy consultando los documentos...",
                "Permíteme buscar en la base de datos..."
            ],
            "not_found": [
                "Lo siento, no encontré información específica sobre eso.",
                "No tengo esa información en mis documentos actuales.",
                "Déjame buscar más detalles sobre tu consulta."
            ],
            "error": [
                "Tuve un problema al procesar tu consulta.",
                "Hubo un error técnico. ¿Podrías intentarlo de nuevo?",
                "No pude procesar tu pregunta correctamente."
            ]
        }
    
    def list_available_voices(self):
        """Lista las voces disponibles"""
        try:
            if not self.voices_path.exists():
                print("No se encontró carpeta de voces")
                return
            
            voices = list(self.voices_path.glob("*.onnx"))
            print(f"\n🎤 VOCES DISPONIBLES ({len(voices)}):")
            
            for i, voice in enumerate(voices, 1):
                print(f"{i}. {voice.stem}")
                
        except Exception as e:
            logger.error(f"Error al listar voces: {e}")
    
    def test_voice(self, voice_model: str = None):
        """Prueba la voz actual o una específica"""
        try:
            if voice_model:
                self.config['voice_model'] = voice_model
                self._initialize_voice()
            
            test_text = f"Hola, soy {self.config.get('assistant_name', 'Udi')}, esta es una prueba de voz con Piper TTS."
            print(f"🔊 Probando voz: {self.config.get('voice_model', 'por defecto')}")
            self.speak(test_text)
            
        except Exception as e:
            logger.error(f"Error en prueba de voz: {e}")

def test_piper_tts():
    """Función de prueba para Piper TTS"""
    print("🚀 PRUEBA DE PIPER TTS")
    print("=" * 50)
    
    try:
        # Inicializar TTS
        tts = PiperTTS()
        
        # Listar voces disponibles
        tts.list_available_voices()
        
        # Probar voz
        tts.test_voice()
        
        print("\n✅ Prueba completada")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_piper_tts() 