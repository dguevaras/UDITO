import logging
import json
import os

def setup_logging():
    """Configurar logging"""
    # Usar ruta relativa al proyecto
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Limpiar archivo de log existente
    log_file = os.path.join(log_dir, 'voice_detector.log')
    if os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write('')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str):
    """Cargar configuración desde archivo JSON"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error al cargar configuración: {e}")
        raise
