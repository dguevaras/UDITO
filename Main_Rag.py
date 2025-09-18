#!/usr/bin/env python3
"""
UDI - Consola RAG (solo texto)
Inicializa el sistema RAG y permite consultas por teclado.
Usa la configuración en config/rag_config.json.
"""

import sys
import json

# Agregar src al path
sys.path.append('src')

from rag.rag_system import RAGSystem


def main():
    print("UDI - Consola RAG (texto)")
    print("=" * 50)
    print("Escribe tu pregunta y presiona Enter. Escribe 'salir' para terminar.")
    print("Usando config: config/rag_config.json\n")

    try:
        rag_system = RAGSystem(config_path="config/rag_config.json")
        rag_system.initialize()

        stats = rag_system.get_stats()
        total_chunks = stats.get('vector_store', {}).get('total_chunks', 0)
        categories = stats.get('vector_store', {}).get('categories', {})
        print(f"Vector store: {total_chunks} chunks en {len(categories)} categorías")

        while True:
            try:
                query = input("\nTu consulta: ").strip()
                if query.lower() in {"salir", "exit", "q", ":q"}:
                    print("\nHasta luego.")
                    break
                if not query:
                    print("Escribe una consulta válida.")
                    continue

                result = rag_system.process_query(query)
                answer = result.get('answer', '')
                source = result.get('source', 'desconocido')
                cls = result.get('classification', {})
                qtype = cls.get('query_type', 'unknown')
                conf = cls.get('confidence', 0.0)

                print("\nRespuesta:")
                print(answer or "(sin respuesta)")
                print(f"\nMeta → fuente: {source} | tipo: {qtype} | confianza: {conf:.2f}")

            except KeyboardInterrupt:
                print("\n\nInterrumpido por usuario.")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Continuando...")

    except Exception as e:
        print(f"Error al iniciar RAG: {e}")


if __name__ == "__main__":
    main()


