from pathlib import Path
import sys

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    print("scikit-learn no está instalado. Instálalo con: pip install scikit-learn")
    sys.exit(1)


def load_documents(path_candidates):
    for p in path_candidates:
        p = Path(p)
        if p.exists():
            text = p.read_text(encoding='utf-8')
            docs = [line.strip() for line in text.splitlines() if line.strip()]
            return docs, p
    raise FileNotFoundError(f"No se encontró el archivo en: {path_candidates}")


def compute_tfidf(docs, stop_words='spanish'):
    vec = TfidfVectorizer(lowercase=True, stop_words=stop_words)
    X = vec.fit_transform(docs)
    return X, vec


if __name__ == '__main__':
    candidates = [
        '01_corpus_turismo_500.txt',
        'data/01_corpus_turismo_500.txt',
        'Data/01_corpus_turismo_500.txt'
    ]

    try:
        docs, used_path = load_documents(candidates)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    X, vectorizer = compute_tfidf(docs)
    feature_names = vectorizer.get_feature_names_out()

    print(f"Archivo leído: {used_path}")
    print(f"Documentos: {X.shape[0]}, Términos (vocab): {X.shape[1]}")

    # Mostrar top términos por los primeros documentos
    import numpy as np
    n_show = min(5, X.shape[0])
    for i in range(n_show):
        row = X[i].toarray().ravel()
        top_idx = row.argsort()[::-1][:10]
        top_terms = [(feature_names[j], float(row[j])) for j in top_idx if row[j] > 0][:10]
        print(f"Doc {i+1} - top términos:")
        for term, score in top_terms:
            print(f"  {term}: {score:.4f}")

    # Opcional: guardar la vocabulario para uso posterior
    import json
    vocab_path = Path('tfidf_vocab.json')
    vocab_path.write_text(json.dumps(feature_names.tolist(), ensure_ascii=False), encoding='utf-8')
    print(f"Vocabulario guardado en: {vocab_path}")

    print("Listo. Ejecuta `python tfidf_step1.py` para repetir el cálculo.")
