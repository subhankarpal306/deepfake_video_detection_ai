import pandas as pd
import networkx as nx


def build_propagation_graph(csv_path):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()

    for _, row in df.iterrows():
        user = row["user_id"]
        post = row["post_id"]

        G.add_node(user, influence=row["influence"], type="user")
        G.add_node(post, type="post")

        G.add_edge(
            user,
            post,
            time=row["time_min"],
            action=row["action"]
        )

    return G
