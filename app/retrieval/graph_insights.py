import re
from .flow_map import load_graph, pretty_name, meta_for


KNOWN_NAMES = {
    "habitsviewmodel": "HabitsViewModel",
    "homeviewmodel": "HomeViewModel",
    "revisionsviewmodel": "RevisionsViewModel",

    "authrepository": "AuthRepository",
    "habitrepository": "HabitRepository",
    "settingsrepository": "SettingsRepository",
    "pushtokenrepository": "PushTokenRepository",

    "credentialmanager": "CredentialManager",
    "firebasemessagingservice": "FirebaseMessagingService",
    "destinyfirebasemessagingservice": "DestinyFirebaseMessagingService",
    "pushtokensyncmanager": "PushTokenSyncManager",
    "remindernotificationmanager": "ReminderNotificationManager",
    "reminderschedulemanager": "ReminderScheduleManager",
    "reminderscheduler": "ReminderScheduler",
}


def node_id_of(node) -> str:
    return str(node.get("id") or node.get("label") or node.get("name") or "")


def smart_name_from_node(node_id: str) -> str:
    parts = node_id.split("_")

    # Prefer duplicate class pattern:
    # push_pushtokensyncmanager_pushtokensyncmanager
    # ui_homescreen_homescreen
    if len(parts) >= 2 and parts[-1] == parts[-2]:
        raw = parts[-1]
    else:
        raw = parts[-1]

    if raw in KNOWN_NAMES:
        return KNOWN_NAMES[raw]

    name = pretty_name(node_id)

    fixes = {
        "Habitrepository": "HabitRepository",
        "Authrepository": "AuthRepository",
        "Settingsrepository": "SettingsRepository",
        "Pushtokenrepository": "PushTokenRepository",
        "Habitsscreen": "HabitsScreen",
        "Homescreen": "HomeScreen",
        "Loginscreen": "LoginScreen",
        "Revisionsscreen": "RevisionsScreen",
        "Settingsscreen": "SettingsScreen",
    }

    return fixes.get(name, name)


def payload(node_id: str, links):
    source_file, source_location = meta_for(node_id, links)

    return {
        "name": smart_name_from_node(node_id),
        "node": node_id,
        "source_file": source_file,
        "source_location": source_location,
    }


def detect_screens(nodes, links):
    results = {}

    for node in nodes:
        node_id = node_id_of(node)

        match = re.match(r"^ui_([a-z0-9]+screen)_\1$", node_id)

        if match:
            results[node_id] = payload(node_id, links)

    return sorted(results.values(), key=lambda x: x["name"])


def detect_viewmodels(nodes, links):
    results = {}

    for node in nodes:
        node_id = node_id_of(node)

        match = re.match(r"^viewmodel_([a-z0-9]+viewmodel)_\1$", node_id)

        if match:
            results[node_id] = payload(node_id, links)

    return sorted(results.values(), key=lambda x: x["name"])


def detect_repositories(nodes, links):
    results = {}

    for node in nodes:
        node_id = node_id_of(node)

        match = re.match(r"^(data|push)_([a-z0-9]+repository)_\2$", node_id)

        if match:
            results[node_id] = payload(node_id, links)

    return sorted(results.values(), key=lambda x: x["name"])


def detect_managers_and_services(nodes, links):
    results = {}

    for node in nodes:
        node_id = node_id_of(node)
        parts = node_id.split("_")

        # Keep only class-level nodes, not methods/helper references.
        # Examples:
        # push_pushtokensyncmanager_pushtokensyncmanager
        # reminder_reminderscheduler_reminderscheduler
        # reminder_remindernotificationmanager_remindernotificationmanager
        if len(parts) >= 2 and parts[-1] == parts[-2]:
            raw = parts[-1]
            if raw.endswith(("manager", "scheduler", "service")):
                results[node_id] = payload(node_id, links)

    return sorted(results.values(), key=lambda x: x["name"])


def repo_summary_dynamic():
    nodes, links = load_graph()

    return {
        "nodes": len(nodes),
        "relations": len(links),
        "screens": detect_screens(nodes, links),
        "viewmodels": detect_viewmodels(nodes, links),
        "repositories": detect_repositories(nodes, links),
        "managers_and_services": detect_managers_and_services(nodes, links),
    }
