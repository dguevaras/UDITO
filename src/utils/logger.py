import logging
import os
from pathlib import Path
import json
from datetime import datetime

def setup_logging(log_level: str = "INFO"):
    """
    Configurar el sistema de logging
    
    Args:
        log_level: Nivel de logging (INFO, DEBUG, ERROR, etc.)
    """
    # Crear directorio de logs si no existe
    logs_dir = Path("data/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar formato de logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configurar logging
    logging.basicConfig(
        level=logging.getLevelName(log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(logs_dir / f"udi_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Crear logger raíz
    logger = logging.getLogger("UDI")
    logger.setLevel(logging.getLevelName(log_level))
    
    # Añadir handlers específicos
    file_handler = logging.FileHandler(logs_dir / f"udi_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler.setFormatter(logging.Formatter(log_format))
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(log_format))
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

def get_logger(name: str):
    """
    Obtener un logger específico
    
    Args:
        name: Nombre del logger
        
    Returns:
        Logger configurado
    """
    return logging.getLogger(name)

def log_error(error: Exception, module: str):
    """
    Registrar un error
    
    Args:
        error: Excepción
        module: Nombre del módulo donde ocurrió el error
    """
    logger = get_logger(module)
    logger.error(f"Error in {module}: {str(error)}")
    logger.error(f"Traceback: {error.__traceback__}")

def log_info(message: str, module: str):
    """
    Registrar información
    
    Args:
        message: Mensaje a registrar
        module: Nombre del módulo
    """
    logger = get_logger(module)
    logger.info(message)

def log_debug(message: str, module: str):
    """
    Registrar debug
    
    Args:
        message: Mensaje a registrar
        module: Nombre del módulo
    """
    logger = get_logger(module)
    logger.debug(message)
