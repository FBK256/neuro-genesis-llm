from mylm_omni.memory import MemoryStore


def test_hybrid_retrieval_finds_semantic_match_without_shared_words(tmp_path):
    def deterministic_embedder(text):
        # A deterministic stand-in, not a real semantic model; verifies the vector-search plumbing.
        words = set(text.casefold().split())
        if words & {"car", "automobile", "vehicle"}:
            return [1.0, 0.0, 0.0]
        if words & {"cat", "feline", "kitten"}:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    store = MemoryStore(
        tmp_path / "memory.sqlite",
        embedder=deterministic_embedder,
        min_semantic_similarity=0.5,
    )
    car = store.add(
        memory_type="semantic",
        key="transport",
        content="A car is a vehicle.",
        source="verified source",
    )
    store.add(
        memory_type="semantic",
        key="animal",
        content="A cat is a feline.",
        source="verified source",
    )
    results = store.retrieve("automobile", limit=2)
    assert results[0]["id"] == car["id"]
    assert results[0]["lexical_overlap"] == 0.0
    assert results[0]["semantic_similarity"] > 0.99


def test_vector_dimension_mismatch_is_rejected(tmp_path):
    def changing_embedder(text):
        return [1.0, 0.0] if "first" in text else [1.0, 0.0, 0.0]

    store = MemoryStore(tmp_path / "memory.sqlite", embedder=changing_embedder)
    store.add(
        memory_type="semantic", key="first", content="first fact", source="source"
    )
    try:
        store.retrieve("different query")
    except ValueError as exc:
        assert "dimension" in str(exc)
    else:
        raise AssertionError("expected embedding dimension mismatch to be rejected")
