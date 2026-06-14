import math
import re

from .flow_map import meta_for


KNOWN_NAMES = {
    "habitsscreen": "HabitsScreen",
    "homescreen": "HomeScreen",
    "loginscreen": "LoginScreen",
    "revisionsscreen": "RevisionsScreen",
    "settingsscreen": "SettingsScreen",

    "habitsviewmodel": "HabitsViewModel",
    "homeviewmodel": "HomeViewModel",
    "revisionsviewmodel": "RevisionsViewModel",

    "authrepository": "AuthRepository",
    "habitrepository": "HabitRepository",
    "settingsrepository": "SettingsRepository",
    "pushtokenrepository": "PushTokenRepository",

    "destinyapp": "DestinyApp",
    "destinyapplication": "DestinyApplication",
    "firebaseruntimeconfig": "FirebaseRuntimeConfig",
    "firebaseauth": "FirebaseAuth",
    "firebasefirestore": "FirebaseFirestore",
    "firebasemessagingservice": "FirebaseMessagingService",
    "destinyfirebasemessagingservice": "DestinyFirebaseMessagingService",

    "credentialmanager": "CredentialManager",
    "pushtokensyncmanager": "PushTokenSyncManager",
    "remindernotificationmanager": "ReminderNotificationManager",
    "reminderschedulemanager": "ReminderScheduleManager",
    "reminderscheduler": "ReminderScheduler",

    "searchbar": "SearchBar",
    "revisionsearchbar": "RevisionSearchBar",
    "flippablehabitcard": "FlippableHabitCard",
    "flippablerevisioncard": "FlippableRevisionCard",
    "habitstatscard": "HabitStatsCard",
    "habitrow": "HabitRow",
    "statcard": "StatCard",
    "progressstatcard": "ProgressStatCard",
    "revisioncard": "RevisionCard",
    "revisiontopiccard": "RevisionTopicCard",
    "habitmilestoneoptionsdialog": "HabitMilestoneOptionsDialog",
    "revisioncompletionoptionsdialog": "RevisionCompletionOptionsDialog",

    "habitdao": "HabitDao",
    "habitentity": "HabitEntity",
    "habitcompletionentity": "HabitCompletionEntity",
    "habitdocument": "HabitDocument",
    "habithistory": "HabitHistory",
    "habituistate": "HabitUiState",
    "findhabitdueatinwindow": "findHabitDueAtInWindow",
}


METHOD_NAMES = {
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

    "acknowledgehabitthirtydaydialog": "acknowledgeHabitThirtyDayDialog",
    "acknowledgerevisioncompletiondialog": "acknowledgeRevisionCompletionDialog",
    "autoresetrevisiontopicifmissed": "autoResetRevisionTopicIfMissed",
    "buildrevisiondaystates": "buildRevisionDayStates",
    "calculatehabitdueat": "calculateHabitDueAt",
    "calculatehabithistory": "calculateHabitHistory",
    "clearall": "clearAll",
    "computedisplayedhabitstreak": "computeDisplayedHabitStreak",
    "computehabitstreak": "computeHabitStreak",
    "getrevisiontopicfortoday": "getRevisionTopicForToday",
    "getrevisiontopicswithprogress": "getRevisionTopicsWithProgress",
    "deletealldocuments": "deleteAllDocuments",
}


NOISE_TERMS = {
    "string", "int", "long", "boolean", "list", "set", "flow", "t", "com",
    "modifier", "context", "activity", "throwable", "stateflow", "job", "class"
}


def node_id_of(node) -> str:
    return str(node.get("id") or node.get("label") or node.get("name") or "")


def compact(value: str) -> str:
    return value.lower().replace("_", "").replace("-", "").replace(".", "")


def to_camel(raw: str) -> str:
    raw = compact(raw)

    if raw in METHOD_NAMES:
        return METHOD_NAMES[raw]

    return raw


def is_noise_node(node_id: str) -> bool:
    parts = node_id.lower().split("_")
    last = parts[-1] if parts else ""

    if last in NOISE_TERMS:
        return True

    if "_kt_" in node_id.lower() and last in NOISE_TERMS:
        return True

    # Example: app_src_main_java_..._kt_long
    if node_id.lower().startswith("app_src_") and any(node_id.lower().endswith("_" + term) for term in NOISE_TERMS):
        return True

    return False


def owner_from_parts(parts):
    for part in reversed(parts):
        raw = compact(part)

        if raw in KNOWN_NAMES and raw.endswith(("screen", "viewmodel", "repository", "manager", "scheduler", "service")):
            return KNOWN_NAMES[raw]

        if raw.endswith("screen"):
            return raw[:-6].capitalize() + "Screen"

        if raw.endswith("viewmodel"):
            return raw[:-9].capitalize() + "ViewModel"

        if raw.endswith("repository"):
            return raw[:-10].capitalize() + "Repository"

    return None


def readable_name(node_id: str) -> str:
    parts = node_id.split("_")
    raw = compact(parts[-1]) if parts else compact(node_id)

    # Class-level duplicate pattern:
    # ui_habitsscreen_habitsscreen
    # data_habitrepository_habitrepository
    if len(parts) >= 2 and compact(parts[-1]) == compact(parts[-2]):
        if raw in KNOWN_NAMES:
            return KNOWN_NAMES[raw]

    if raw in KNOWN_NAMES:
        return KNOWN_NAMES[raw]

    # Method-level pattern:
    # data_habitrepository_habitrepository_addhabit
    # viewmodel_habitsviewmodel_habitsviewmodel_addhabit
    owner = owner_from_parts(parts[:-1])
    if owner and raw not in NOISE_TERMS:
        return f"{owner}.{to_camel(raw)}"

    for part in reversed(parts):
        part_compact = compact(part)
        if part_compact in KNOWN_NAMES:
            return KNOWN_NAMES[part_compact]

    if raw.endswith("screen"):
        return raw[:-6].capitalize() + "Screen"

    if raw.endswith("viewmodel"):
        return raw[:-9].capitalize() + "ViewModel"

    if raw.endswith("repository"):
        return raw[:-10].capitalize() + "Repository"

    if raw.endswith("manager"):
        return raw[:-7].capitalize() + "Manager"

    if raw.endswith("scheduler"):
        return raw[:-9].capitalize() + "Scheduler"

    if raw.endswith("service"):
        return raw[:-7].capitalize() + "Service"

    return raw


def relation_phrase(link: dict) -> str:
    relation = link.get("relation") or ""
    context = link.get("context") or ""

    if relation == "contains":
        return "contains"
    if relation == "calls":
        return "calls"
    if relation == "method":
        return "defines method"
    if relation == "inherits":
        return "extends"
    if relation == "references":
        if context == "parameter_type":
            return "uses as parameter"
        if context == "return_type":
            return "returns"
        if context == "field":
            return "has field"
        if context == "generic_arg":
            return "uses generic type"
        return "references"

    return relation or "related to"


def format_link(link: dict) -> dict:
    source = link.get("source") or ""
    target = link.get("target") or ""

    return {
        "source": source,
        "source_name": readable_name(source),
        "target": target,
        "target_name": readable_name(target),
        "relation": link.get("relation"),
        "relation_label": relation_phrase(link),
        "context": link.get("context"),
        "source_file": link.get("source_file"),
        "source_location": link.get("source_location"),
    }


def node_payload(node_id: str, links: list[dict]) -> dict:
    source_file, source_location = meta_for(node_id, links)

    return {
        "name": readable_name(node_id),
        "node": node_id,
        "source_file": source_file,
        "source_location": source_location,
    }


def is_architecture_node(name: str) -> bool:
    c = compact(name)
    return any(x in c for x in [
        "screen", "viewmodel", "repository", "manager", "scheduler", "service", "config"
    ])


def is_class_level_node(node_id: str) -> bool:
    parts = [compact(p) for p in node_id.split("_")]
    return len(parts) >= 2 and parts[-1] == parts[-2]


def search_nodes(query: str, nodes: list[dict], links: list[dict], limit: int = 30) -> list[dict]:
    q = compact(query)
    best_by_name = {}

    for node in nodes:
        node_id = node_id_of(node)

        if is_noise_node(node_id):
            continue

        name = readable_name(node_id)
        node_compact = compact(node_id)
        name_compact = compact(name)

        score = 0
        matched = False

        if q == name_compact:
            score += 120
            matched = True
        elif q in name_compact:
            score += 90
            matched = True

        if q == node_compact:
            score += 80
            matched = True
        elif q in node_compact:
            score += 45
            matched = True

        # Do not include architecture nodes unless they actually match the query.
        if not matched:
            continue

        if is_architecture_node(name):
            score += 25

        if is_class_level_node(node_id):
            score += 30

        # Prefer cleaner display names over repeated raw method noise.
        if "." in name:
            score -= 5

        if score <= 0:
            continue

        # For method-level results, avoid returning unrelated methods just because
        # the owner matches the query. Example: query "habit" should not rank
        # addRevisionTopic highly only because it lives inside HabitRepository.
        if "." in name:
            method_part = name.split(".")[-1]
            if q not in compact(method_part):
                score -= 65

        if score <= 0:
            continue

        payload = node_payload(node_id, links)
        payload["score"] = score

        existing = best_by_name.get(payload["name"])
        if existing is None or payload["score"] > existing["score"]:
            best_by_name[payload["name"]] = payload

    results = list(best_by_name.values())
    results.sort(key=lambda item: (-item["score"], item["name"]))
    return results[:limit]


_WORD_SPLIT_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def word_tokens(node_id: str, name: str) -> set:
    """Best-effort set of word tokens for a node, from its camelCase display
    name and underscore-separated id. Used to prefer real word-boundary matches
    (so "mode" matches deleteMode but not the "model" inside ViewModel)."""
    tokens = set()

    for chunk in re.split(r"[^A-Za-z0-9]+", name):
        for word in _WORD_SPLIT_RE.findall(chunk):
            if len(word) >= 2:
                tokens.add(word.lower())

    for part in node_id.split("_"):
        part = part.strip().lower()
        if len(part) >= 2:
            tokens.add(part)

    return tokens


def rank_nodes_for_query(
    query_terms: list[str],
    nodes: list[dict],
    links: list[dict],
    limit: int = 30,
    boosts: dict = None,
) -> list[dict]:
    """Rank nodes against a multi-word query.

    Each query term is weighted by rarity (inverse document frequency) so a
    specific word like "strict" counts far more than a generic one like "mode"
    or "app". A node that covers several distinct query terms gets a bonus, so
    nodes actually about the question float to the top instead of being crowded
    out by common-substring noise.

    `boosts` is an optional per-workspace {term: multiplier} map (a tunable
    RetrievalConfig knob); a term's IDF weight is scaled by its multiplier so
    admins can amplify domain-important words without code changes.
    """
    terms = []
    seen_terms = set()
    for term in query_terms:
        compacted = compact(term)
        if compacted and compacted not in seen_terms:
            seen_terms.add(compacted)
            terms.append(compacted)

    if not terms:
        return []

    # Boost keys are compacted to match how terms are normalized above.
    boost_by_term = {compact(k): float(v) for k, v in (boosts or {}).items()}

    # Pass 1: collect candidates and per-term document frequency.
    candidates = []
    doc_freq = {term: 0 for term in terms}

    for node in nodes:
        node_id = node_id_of(node)

        if is_noise_node(node_id):
            continue

        name = readable_name(node_id)
        name_compact = compact(name)
        node_compact = compact(node_id)
        words = word_tokens(node_id, name)

        matched = {}
        for term in terms:
            if term in words or term == name_compact:
                matched[term] = 1.0          # clean word / exact-name match
            elif term in name_compact:
                matched[term] = 0.55         # substring of the display name
            elif term in node_compact:
                matched[term] = 0.35         # substring of the raw id only

        if matched:
            for term in matched:
                doc_freq[term] += 1
            candidates.append((node_id, name, matched))

    total = max(1, len(candidates))
    weight = {
        term: (math.log(1 + total / freq) if freq else 0.0) * boost_by_term.get(term, 1.0)
        for term, freq in doc_freq.items()
    }

    best_by_name = {}
    for node_id, name, matched in candidates:
        base = sum(weight[term] * quality for term, quality in matched.items())
        if base <= 0:
            continue

        coverage = 1.0 + 0.6 * (len(matched) - 1)
        score = base * coverage * 100.0

        # Structural preferences, kept secondary to relevance.
        if is_architecture_node(name):
            score += 15
        if is_class_level_node(node_id):
            score += 20
        if "." in name:
            score -= 4

        payload = node_payload(node_id, links)
        payload["score"] = round(score, 2)

        existing = best_by_name.get(payload["name"])
        if existing is None or payload["score"] > existing["score"]:
            best_by_name[payload["name"]] = payload

    results = list(best_by_name.values())
    results.sort(key=lambda item: (-item["score"], item["name"]))
    return results[:limit]
