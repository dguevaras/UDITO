#!/usr/bin/env python3
"""
UDI - Main Consola STT → RAG/NLP → TTS
Sistema de consola para preguntas escritas con respuesta por voz
"""

import sys
import os
from pathlib import Path

# Agregar src al path
sys.path.append('src')

# Importar componentes existentes
from rag.rag_system import RAGSystem
from tts.piper_tts_real import PiperTTS

def main():
    """Función principal del sistema de consola"""
    print("🎓 UDI - Sistema de Consultas Universitarias")
    print("=" * 50)
    print("📝 Escribe tus preguntas y UDI responderá por voz")
    print("🔍 Sistema RAG/NLP + Piper TTS")
    print("⏹️  Escribe 'salir' para terminar")
    print("=" * 50)
    
    try:
        # Inicializar sistema RAG
        print("🔄 Inicializando sistema RAG...")
        rag_system = RAGSystem()
        rag_system.initialize()
        print("✅ Sistema RAG inicializado")
        
        # Inicializar TTS
        print("🔊 Inicializando Piper TTS...")
        tts = PiperTTS()
        print("✅ Piper TTS inicializado")
        
        # Mostrar estadísticas del sistema
        stats = rag_system.get_stats()
        print(f"\n📊 Sistema listo:")
        print(f"   - Chunks procesados: {stats.get('vector_store', {}).get('total_chunks', 0)}")
        print(f"   - Categorías: {len(stats.get('vector_store', {}).get('categories', []))}")
        print(f"   - Voz: {tts.config.get('voice_model', 'por defecto')}")
        
        print("\n" + "=" * 50)
        print("🎤 ¡Listo para responder tus preguntas!")
        print("=" * 50)
        
        # Bucle principal
        while True:
            try:
                # Obtener pregunta del usuario
                pregunta = input("\n❓ Tu pregunta: ").strip()
                
                # Verificar salida
                if pregunta.lower() in ['salir', 'exit', 'quit', 'q']:
                    print("\n👋 ¡Gracias por usar UDI!")
                    break
                
                if not pregunta:
                    print("⚠️  Por favor, escribe una pregunta")
                    continue
                
                print(f"\n🔍 Procesando: '{pregunta}'")
                
                # Procesar con RAG
                respuesta = rag_system.process_query(pregunta)
                
                # Mostrar clasificación
                clasificacion = respuesta.get('classification', {})
                tipo_consulta = clasificacion.get('query_type', 'unknown')
                confianza = clasificacion.get('confidence', 0.0)
                
                print(f"📊 Clasificación: {tipo_consulta} (confianza: {confianza:.2f})")
                print(f"🔗 Fuente: {respuesta.get('source', 'unknown')}")
                
                # Mostrar respuesta
                print(f"\n💬 Respuesta:")
                print(f"   {respuesta.get('answer', 'No se pudo generar respuesta')}")
                
                # Reproducir con TTS
                print(f"\n🔊 Reproduciendo respuesta...")
                tts.speak(respuesta.get('answer', 'No se pudo generar respuesta'))
                
                print("✅ Respuesta completada")
                
            except KeyboardInterrupt:
                print("\n\n👋 ¡Gracias por usar UDI!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("🔄 Continuando...")
                
    except Exception as e:
        print(f"❌ Error al inicializar el sistema: {e}")
        print("🔧 Verifica que todos los componentes estén disponibles")

if __name__ == "__main__":
    main()
