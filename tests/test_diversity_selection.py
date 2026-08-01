from evo_prompt.model import GenerationRecord, select_diverse_parents

def test_select_diverse_parents_logic():
    """
    Verifies that the optimizer picks the best prompt AND the most different prompt,
    even if the most different prompt has a lower score than the 2nd best.
    """
    embeddings = [
        [1.0, 0.0],     # p0: "Cyberpunk girl"
        [0.99, 0.01],   # p1: "Cyberpunk girl in blue" (Very similar to p0)
        [0.0, 1.0],     # p2: "Renaissance painting" (Very different)
        [0.01, 0.99],   # p3: "Renaissance oil painting" (Very similar to p2)
    ]

    records = [
        GenerationRecord(
            intent="test",
            prompt=f"prompt_{i}",
            image_path=f"img_{i}.png",
            rubric={"score": 2.0},
            scalar_score=[0.95, 0.90, 0.85, 0.80][i],
            generation=0,
            embedding=embeddings[i],
            prompt_reasoning="",
            scoring_reasoning=""
        )
        for i in range(4)
    ]

    selected = select_diverse_parents(records, top_k=2, similarity_threshold=0.85)

    # Should pick index 0 (best score) and index 2 (most different, despite lower score than idx 1)
    assert len(selected) == 2
    assert selected[0].prompt == "prompt_0"
    assert selected[1].prompt == "prompt_2"
