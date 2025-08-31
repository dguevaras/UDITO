import json
import logging
from pathlib import Path

def setup_logging():
    """Configurar logging"""
    try:
        # Crear directorio de logs si no existe
        logs_dir = Path(__file__).parent.parent / 'logs'
        logs_dir.mkdir(exist_ok=True)
        
        # Crear archivo de log
        log_file = logs_dir / 'voice_detector.log'
        
        # Configurar logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    except Exception as e:
        print(f"Error configurando logging: {e}")
        raise

def load_config(config_path: str = "settings.json") -> dict:
    """Cargar y validar la configuración"""
    try:
        # Obtener ruta absoluta del archivo de configuración en el directorio src
        config_file = Path(__file__).parent.parent / config_path
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Error: El archivo de configuración no existe en {config_file}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error: El archivo de configuración no es un JSON válido")
        raise
    except Exception as e:
        logging.error(f"Error inesperado al cargar la configuración: {e}")
        raise
