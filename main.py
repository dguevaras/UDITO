#!/usr/bin/env python3
"""
UDI - Sistema principal simple
Pipeline Unificados - Openwakeword + Whisper + RAG/NLP + TTS + Piper + handler de audio unificado
"""

import logging
import time
from src.wake_word import MycroftDetector

# Configurar logging simple
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UDI")

def on_wake_word():
    """Callback cuando se detecta UDITO"""
    print("🎉 ¡UDITO detectado!")
    print("🔔 Activando sistema...")
    time.sleep(0.1)  # Simular activación rápida
    print("✅ Sistema activo")

def main():
    """Función principal simple"""
    print("🎧 UDI - Solo Mycroft Precise")
    print("🔊 Wake Word: UDITO")
    print("⏹️  Ctrl+C para salir")
    
    try:
        detector = MycroftDetector("udito_model.net")
        detector.start_listening(on_wake_word)
    except KeyboardInterrupt:
        print("\n👋 UDI detenido")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()
