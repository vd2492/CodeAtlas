import json
import re
from pathlib import Path

GRAPH_PATH = Path("graphify-out/graph.json")


def title_case_node(value: str) -> str:
    parts = re.split(r"[_\-\s]+", value)
    clean = []
    for p in parts:
        if not p:
            continue
        clean.append(p[:1].upper() + p[1:])
    return "".join(clean)


def get_node_id(node) -> str:
    return str(node.get("id") or node.get("label") or node.get("name") or "")


def load_graph():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError("graphify-out/graph.json not found. Run: graphify update . --no-cluster")

    graph = json.loads(GRAPH_PATH.read_text())

    # Graphify may use different keys depending on version/output mode.
    nodes = graph.get("nodes", [])
    edges = graph.get("edges") or graph.get("links") or graph.get("relationships") or []

    return graph, nodes, edges


def detect_main_screens(nodes):
    screens = {}

    for node in nodes:
        node_id = get_node_id(node)

        # Best signal for Compose screen function:
        # ui_homescreen_homescreen
        # ui_loginscreen_loginscreen
        match = re.match(r"^ui_([a-z0-9]+screen)_\1$", node_id)
        if match:
            raw = match.group(1)
            screens[raw] = {
                "name": title_case_node(raw),
                "node": node_id,
                "type": "Composable Screen"
            }

    return list(screens.values())


def detect_viewmodels(nodes):
    viewmodels = {}

    for node in nodes:
        node_id = get_node_id(node)

        # Best signal:
        # viewmodel_homeviewmodel_homeviewmodel
        match = re.match(r"^viewmodel_([a-z0-9]+viewmodel)_\1$", node_id)
        if match:
            raw = match.group(1)
            viewmodels[raw] = {
                "name": title_case_node(raw),
                "node": node_id
            }

    return list(viewmodels.values())


def detect_repositories(nodes):
    repositories = {}

    for node in nodes:
        node_id = get_node_id(node)

        # Best signal:
        # data_habitrepository_habitrepository
        # data_authrepository_authrepository
        match = re.match(r"^(data|push)_([a-z0-9]+repository)_\2$", node_id)
        if match:
            raw = match.group(2)
            repositories[raw] = {
                "name": title_case_node(raw),
                "node": node_id
            }

    return list(repositories.values())


def detect_feature_methods(nodes):
    important_keywords = [
        "addhabit",
        "deletehabit",
        "togglehabit",
        "restarthabit",
        "continuehabitstreak",
        "addtopic",
        "deletetopic",
        "startrevision",
        "completerevision",
        "restartrevision",
        "login",
        "logout",
        "register",
        "synccurrenttoken",
        "setstrictmodeenabled",
    ]

    methods = []

    for node in nodes:
        node_id = get_node_id(node)
        compact = node_id.lower().replace("_", "")

        if any(keyword in compact for keyword in important_keywords):
            if not any(skip in compact for skip in ["boolean", "string", "int", "long", "stateflow", "modifier"]):
                methods.append(node_id)

    return sorted(set(methods))


def print_section(title, rows):
    print(f"\n=== {title} ===")
    if not rows:
        print("- None detected")
        return

    for row in rows:
        if isinstance(row, dict):
            print(f"- {row['name']}  [{row['node']}]")
        else:
            print(f"- {row}")


def main():
    graph, nodes, edges = load_graph()

    screens = detect_main_screens(nodes)
    viewmodels = detect_viewmodels(nodes)
    repositories = detect_repositories(nodes)
    feature_methods = detect_feature_methods(nodes)

    print("\n=== CodeAtlas Clean Repo Summary ===")
    print(f"Total nodes: {len(nodes)}")
    print(f"Total relations found: {len(edges)}")
    print(f"Graph keys: {', '.join(graph.keys())}")

    print_section("Main Screens", screens)
    print_section("ViewModels", viewmodels)
    print_section("Repositories", repositories)
    print_section("Important Feature Methods", feature_methods[:40])


if __name__ == "__main__":
    main()
