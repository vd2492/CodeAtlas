import json
import re
import sys
from pathlib import Path

GRAPH_PATH = Path("graphify-out/graph.json")


def get_node_id(node) -> str:
    return str(node.get("id") or node.get("label") or node.get("name") or "")


def clean_name(raw: str) -> str:
    raw = raw.replace("_", " ")
    words = raw.split()
    final = []

    for word in words:
        lower = word.lower()

        if lower.endswith("screen"):
            base = lower[:-6]
            final.append(base.capitalize() + "Screen")
        elif lower.endswith("viewmodel"):
            base = lower[:-9]
            final.append(base.capitalize() + "ViewModel")
        elif lower.endswith("repository"):
            base = lower[:-10]
            final.append(base.capitalize() + "Repository")
        else:
            final.append(word.capitalize())

    return " ".join(final)


def load_graph():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError("graphify-out/graph.json not found. Run: graphify update . --no-cluster")

    graph = json.loads(GRAPH_PATH.read_text())
    return graph, graph.get("nodes", []), graph.get("links", [])


def detect_main_screens(nodes):
    results = {}

    for node in nodes:
        node_id = get_node_id(node)
        match = re.match(r"^ui_([a-z0-9]+screen)_\1$", node_id)

        if match:
            raw = match.group(1)
            results[raw] = {
                "name": clean_name(raw),
                "node": node_id,
            }

    return list(results.values())


def detect_viewmodels(nodes):
    results = {}

    for node in nodes:
        node_id = get_node_id(node)
        match = re.match(r"^viewmodel_([a-z0-9]+viewmodel)_\1$", node_id)

        if match:
            raw = match.group(1)
            results[raw] = {
                "name": clean_name(raw),
                "node": node_id,
            }

    return list(results.values())


def detect_repositories(nodes):
    results = {}

    for node in nodes:
        node_id = get_node_id(node)
        match = re.match(r"^(data|push)_([a-z0-9]+repository)_\2$", node_id)

        if match:
            raw = match.group(2)
            results[raw] = {
                "name": clean_name(raw),
                "node": node_id,
            }

    return list(results.values())


def find_nodes_containing(nodes, keywords):
    matches = []

    for node in nodes:
        node_id = get_node_id(node)
        compact = node_id.lower().replace("_", "")

        if any(keyword in compact for keyword in keywords):
            if not any(skip in compact for skip in ["boolean", "string", "int", "long", "modifier", "stateflow"]):
                matches.append(node_id)

    return sorted(set(matches))


def answer_main_screens(nodes):
    screens = detect_main_screens(nodes)

    print("\nCodeAtlas Answer: Main screens detected\n")

    for screen in screens:
        print(f"- {screen['name']}")
        print(f"  node: {screen['node']}")

    print("\nStakeholder summary:")
    print("This app appears to have screens for login, home/dashboard, habits, revisions/spaced repetition, and settings.")


def answer_viewmodels(nodes):
    viewmodels = detect_viewmodels(nodes)

    print("\nCodeAtlas Answer: ViewModels detected\n")

    for vm in viewmodels:
        print(f"- {vm['name']}")
        print(f"  node: {vm['node']}")

    print("\nStakeholder summary:")
    print("The app follows an MVVM-like structure where screens are backed by ViewModels for habits, home, and revisions.")


def answer_repositories(nodes):
    repositories = detect_repositories(nodes)

    print("\nCodeAtlas Answer: Repositories detected\n")

    for repo in repositories:
        print(f"- {repo['name']}")
        print(f"  node: {repo['node']}")

    print("\nStakeholder summary:")
    print("The app has repositories for authentication, habit/revision data, settings, and push notification tokens.")


def answer_habit_flow(nodes):
    keywords = [
        "habitsscreen",
        "habitsviewmodel",
        "habitrepository",
        "addhabit",
        "deletehabit",
        "togglehabit",
        "restarthabit",
        "continuehabitstreak",
        "habitwithstats",
        "habitcompletionstate",
    ]

    matches = find_nodes_containing(nodes, keywords)

    print("\nCodeAtlas Answer: Habit flow\n")
    print("Likely flow:")
    print("1. User interacts with HabitsScreen or HomeScreen.")
    print("2. UI calls HabitsViewModel or HomeViewModel.")
    print("3. ViewModel calls HabitRepository.")
    print("4. HabitRepository handles habit creation, deletion, streaks, completion state, and persistence.")

    print("\nRelevant graph nodes:")
    for node in matches[:40]:
        print(f"- {node}")


def answer_revision_flow(nodes):
    keywords = [
        "revisionsscreen",
        "revisionsviewmodel",
        "habitrepository",
        "revision",
        "addtopic",
        "deletetopic",
        "startrevision",
        "completerevision",
        "restartrevision",
        "revisiontopic",
        "revisionday",
    ]

    matches = find_nodes_containing(nodes, keywords)

    print("\nCodeAtlas Answer: Revision / spaced repetition flow\n")
    print("Likely flow:")
    print("1. User interacts with RevisionsScreen or HomeScreen.")
    print("2. UI calls RevisionsViewModel or HomeViewModel.")
    print("3. ViewModel calls HabitRepository.")
    print("4. HabitRepository manages revision topics, revision days, due dates, completion, restart, and missed revision logic.")

    print("\nRelevant graph nodes:")
    for node in matches[:50]:
        print(f"- {node}")


def answer_login_flow(nodes):
    keywords = [
        "loginscreen",
        "authrepository",
        "login",
        "logout",
        "register",
        "googleidtoken",
        "credential",
        "firebase",
        "saveuserprofile",
    ]

    matches = find_nodes_containing(nodes, keywords)

    print("\nCodeAtlas Answer: Login/auth flow\n")
    print("Likely flow:")
    print("1. User interacts with LoginScreen.")
    print("2. LoginScreen handles email/password or Google sign-in.")
    print("3. AuthRepository performs login/register/logout logic.")
    print("4. User profile is saved or fetched through Firebase-related code.")

    print("\nRelevant graph nodes:")
    for node in matches[:50]:
        print(f"- {node}")


def route_question(question, nodes):
    q = question.lower()

    if "screen" in q:
        answer_main_screens(nodes)
    elif "viewmodel" in q or "view model" in q:
        answer_viewmodels(nodes)
    elif "repository" in q or "data layer" in q:
        answer_repositories(nodes)
    elif "habit" in q:
        answer_habit_flow(nodes)
    elif "revision" in q or "spaced" in q:
        answer_revision_flow(nodes)
    elif "login" in q or "auth" in q or "sign" in q:
        answer_login_flow(nodes)
    else:
        print("\nCodeAtlas could not classify this question yet.")
        print("Try asking about screens, ViewModels, repositories, habit flow, revision flow, or login flow.")


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 codeatlas_tools/ask_codeatlas.py "What are the main screens?"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    graph, nodes, links = load_graph()

    print(f"\nQuestion: {question}")
    print(f"Graph loaded: {len(nodes)} nodes, {len(links)} relations")

    route_question(question, nodes)


if __name__ == "__main__":
    main()
