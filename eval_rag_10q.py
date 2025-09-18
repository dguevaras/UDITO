#!/usr/bin/env python3
"""
Evaluación rápida del RAG con 10 preguntas basadas en documentos.
Guarda resultados en data/eval/rag_eval_YYYYMMDD_HHMMSS.txt
"""

import sys
import os
import json
from datetime import datetime

# Asegurar import de src
sys.path.append('src')

from rag.rag_system import RAGSystem


def main():
    out_dir = os.path.join('data', 'eval')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    # Preguntas orientadas al corpus (evitar FAQ básicas)
    questions = [
        "¿Cuáles son las políticas de calidad y medio ambiente de la UDIT?",
        "¿Qué establece la normativa de acceso y admisión?",
        "¿Qué es la normativa RAC y a qué aplica?",
        "¿Cuáles son las condiciones generales del curso 2024-2025?",
        "¿Cuándo abre y cierra la universidad?",
        "¿Cuál es el horario de la secretaría académica?",
        "¿Qué requisitos existen para la extinción de titulaciones oficiales?",
        "¿Qué documentación se exige para la admisión?",
        "¿Qué servicios están disponibles y cuáles son sus horarios?",
        "¿Existen horarios especiales durante el periodo de matrícula?",
    ]

    rag = RAGSystem(config_path="config/rag_config.json")
    rag.initialize()

    # Desactivar capa de respuestas básicas para esta evaluación
    try:
        rag.basic_qa_manager.qa_data = {"questions": [], "identity_questions": [], "udit_specific": [], "greetings": []}
    except Exception:
        pass

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("Evaluación RAG (10 preguntas)\n")
        f.write("="*60 + "\n\n")
        stats = rag.get_stats()
        f.write("Contexto índice:\n")
        f.write(json.dumps({
            "total_chunks": stats.get('vector_store', {}).get('total_chunks', 0),
            "categories": stats.get('vector_store', {}).get('categories', {}),
            "embedding_model": stats.get('vector_store', {}).get('model_name', '')
        }, ensure_ascii=False, indent=2))
        f.write("\n\n")

        for i, q in enumerate(questions, 1):
            try:
                res = rag.process_query(q)
                answer = (res.get('answer') or '').strip()
                src = res.get('source', 'unknown')
                cls = res.get('classification', {})
                qtype = cls.get('query_type', 'unknown')
                conf = cls.get('confidence', 0.0)
                used_llm = res.get('llm_used', False)
                search_len = len(res.get('search_results', [])) if 'search_results' in res else 'NA'

                f.write(f"[{i}] PREGUNTA: {q}\n")
                f.write(f"     CLASIF: {qtype} (conf: {conf:.2f}) | fuente: {src} | llm: {used_llm} | hits: {search_len}\n")
                f.write("     RESPUESTA:\n")
                # Limitar tamaño en archivo
                snippet = answer if len(answer) < 1200 else answer[:1200] + "..."
                for line in snippet.splitlines():
                    f.write(f"       {line}\n")
                f.write("\n")
            except Exception as e:
                f.write(f"[{i}] ERROR procesando pregunta: {q} -> {e}\n\n")

    print(out_path)


if __name__ == '__main__':
    main()




