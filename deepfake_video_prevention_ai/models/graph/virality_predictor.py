def predict_virality(graph):
    """
    Estimate how likely content is to go viral
    """

    users = [
        n for n, d in graph.nodes(data=True)
        if d.get("type") == "user"
    ]

    if not users:
        return 0.0

    total_influence = sum(
        graph.nodes[u].get("influence", 0)
        for u in users
    )

    spread_speed = len(graph.edges) / len(users)

    virality_score = min(
        1.0,
        0.6 * spread_speed + 0.4 * (total_influence / len(users))
    )

    return virality_score
