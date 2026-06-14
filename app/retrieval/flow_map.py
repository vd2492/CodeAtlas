import json
import re
import sys
from pathlib import Path

from ..config import graph_path

TOPICS = {
    "habit": {
        "title": "Habit flow",
        "screens": ["ui_habitsscreen_habitsscreen", "ui_homescreen_homescreen"],
        "viewmodels": [
            "viewmodel_habitsviewmodel_habitsviewmodel",
            "viewmodel_homeviewmodel_homeviewmodel",
        ],
        "repositories": ["data_habitrepository_habitrepository"],
        "method_keywords": [
            "addhabit",
            "deletehabit",
            "togglehabit",
            "restarthabit",
            "continuehabitstreak",
            "sethabitstatetoday",
            "gettodayhabitswithcompletion",
            "gethabitswithstats",
            "habitsflow",
            "currentuserhabitscollection",
        ],
    },
    "revision": {
        "title": "Revision / spaced repetition flow",
        "screens": ["ui_revisionsscreen_revisionsscreen", "ui_homescreen_homescreen"],
        "viewmodels": [
            "viewmodel_revisionsviewmodel_revisionsviewmodel",
            "viewmodel_homeviewmodel_homeviewmodel",
        ],
        "repositories": ["data_habitrepository_habitrepository"],
        "method_keywords": [
            "addrevisiontopic",
            "deleterevisiontopic",
            "startrevision",
            "completerevision",
            "completeactiverevision",
            "restartrevision",
            "restartrevisiontopic",
            "revisionsflow",
            "currentuserrevisionscollection",
            "calculaterevisiondueat",
            "findmissedrevisionday",
        ],
    },
    "login": {
        "title": "Login / authentication flow",
        "screens": ["ui_loginscreen_loginscreen"],
        "viewmodels": [],
        "repositories": ["data_authrepository_authrepository"],
        "method_keywords": [
            "login",
            "loginwithgoogleidtoken",
            "register",
            "logout",
            "saveuserprofile",
            "requestgoogleidtoken",
            "extractgoogleidtoken",
            "mapgooglesigninerror",
        ],
    },
}


def load_graph(graph_file=None):
    path = Path(graph_file) if graph_file else graph_path()
    if not path.exists():
        raise FileNotFoundError(
            f"graph.json not found at {path}. Index a repo first (graphify update <path> --no-cluster)."
        )

    graph = json.loads(path.read_text())
    return graph.get("nodes", []), graph.get("links", [])


def get_node_id(node):
    return str(node.get("id") or node.get("label") or node.get("name") or "")


def pretty_name(node_id):
    parts = node_id.split("_")

    if node_id.startswith("ui_"):
        screen_parts = [p for p in parts if p.endswith("screen") and p != "screen"]
        if screen_parts:
            base = screen_parts[-1].replace("screen", "")
            return base.capitalize() + "Screen"

    if "viewmodel" in node_id:
        vm_parts = [p for p in parts if p.endswith("viewmodel") and p != "viewmodel"]
        if vm_parts:
            base = vm_parts[-1].replace("viewmodel", "")
            return base.capitalize() + "ViewModel"

    if "repository" in node_id:
        repo_parts = [p for p in parts if p.endswith("repository") and p != "repository"]
        if repo_parts:
            base = repo_parts[-1].replace("repository", "")
            return base.capitalize() + "Repository"

    return node_id


def to_camel_case(raw):
    parts = raw.split("_")
    if len(parts) > 1:
        return parts[0] + "".join(p.capitalize() for p in parts[1:])

    known = {
        "addhabit": "addHabit",
        "deletehabit": "deleteHabit",
        "togglehabit": "toggleHabit",
        "restarthabit": "restartHabit",
        "continuehabitstreak": "continueHabitStreak",
        "currentuserhabitscollection": "currentUserHabitsCollection",
        "gethabitswithstats": "getHabitsWithStats",
        "gettodayhabitswithcompletion": "getTodayHabitsWithCompletion",
        "sethabitstatetoday": "setHabitStateToday",
        "togglehabitalarm": "toggleHabitAlarm",
        "addrevisiontopic": "addRevisionTopic",
        "deleterevisiontopic": "deleteRevisionTopic",
        "startrevision": "startRevision",
        "completerevision": "completeRevision",
        "completeactiverevision": "completeActiveRevision",
        "restartrevisiontopic": "restartRevisionTopic",
        "currentuserrevisionscollection": "currentUserRevisionsCollection",
        "calculaterevisiondueat": "calculateRevisionDueAt",
        "findmissedrevisionday": "findMissedRevisionDay",
        "loginwithgoogleidtoken": "loginWithGoogleIdToken",
        "saveuserprofile": "saveUserProfile",
        "requestgoogleidtoken": "requestGoogleIdToken",
        "extractgoogleidtoken": "extractGoogleIdToken",
        "mapgooglesigninerror": "mapGoogleSignInError",
    }
    return known.get(raw, raw)


def pretty_method(node_id):
    method = node_id.split("_")[-1]
    method = to_camel_case(method)

    owner = None
    for part in node_id.split("_"):
        if part.endswith("repository") and part != "repository":
            base = part.replace("repository", "")
            owner = base.capitalize() + "Repository"
        elif part.endswith("viewmodel") and part != "viewmodel":
            base = part.replace("viewmodel", "")
            owner = base.capitalize() + "ViewModel"

    if owner:
        return f"{owner}.{method}"

    return method

def meta_for(node_id, links):
    for link in links:
        if link.get("source") == node_id or link.get("target") == node_id:
            source_file = link.get("source_file", "unknown")
            source_location = link.get("source_location", "?")
            return source_file, source_location

    return "unknown", "?"


def find_methods(nodes, config):
    method_nodes = []

    for node in nodes:
        node_id = get_node_id(node)
        compact = node_id.lower().replace("_", "")

        if any(keyword in compact for keyword in config["method_keywords"]):
            if not any(skip in compact for skip in ["boolean", "string", "int", "long", "modifier", "stateflow", "list", "flow"]):
                method_nodes.append(node_id)

    return sorted(set(method_nodes))


def print_component(title, node_ids, links):
    print(f"\n{title}")

    for node_id in node_ids:
        source_file, source_location = meta_for(node_id, links)
        print(f"- {pretty_name(node_id)}")
        print(f"  node: {node_id}")
        print(f"  source: {source_file} {source_location}")


def print_flow(topic):
    nodes, links = load_graph()

    if topic not in TOPICS:
        print("Supported topics: habit, revision, login")
        sys.exit(1)

    config = TOPICS[topic]

    print(f"\n=== CodeAtlas Flow Map: {config['title']} ===")
    print(f"Graph loaded: {len(nodes)} nodes, {len(links)} relations")

    print("\nHigh-level flow:")
    if config["viewmodels"]:
        print("Screen → ViewModel → Repository → Data/Persistence")
    else:
        print("Screen → Repository/Auth service → Data/Persistence")

    print_component("Screens", config["screens"], links)

    if config["viewmodels"]:
        print_component("ViewModels", config["viewmodels"], links)

    print_component("Repositories", config["repositories"], links)

    methods = find_methods(nodes, config)

    print("\nImportant methods/actions")
    for node_id in methods[:30]:
        source_file, source_location = meta_for(node_id, links)
        print(f"- {pretty_method(node_id)}")
        print(f"  node: {node_id}")
        print(f"  source: {source_file} {source_location}")

    print("\nStakeholder summary:")

    if topic == "habit":
        print("Users manage habits from HabitsScreen and HomeScreen. The UI talks to HabitsViewModel/HomeViewModel, which use HabitRepository for habit state, streaks, creation, deletion, restart, and persistence.")
    elif topic == "revision":
        print("Users manage revision/spaced-repetition topics from RevisionsScreen and HomeScreen. The ViewModels use HabitRepository for revision topic creation, due-date calculation, completion, restart, and missed-day handling.")
    elif topic == "login":
        print("Users authenticate from LoginScreen. The screen handles login inputs and Google sign-in token flow, while AuthRepository handles login, registration, logout, and profile persistence.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 codeatlas_tools/flow_map.py habit|revision|login")
        sys.exit(1)

    print_flow(sys.argv[1].lower())
