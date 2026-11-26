import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from app import config, db, group_ask
from app.agent.tools import GroupRepositoryToolbox
from app.repos import group_routes, groups


class GroupTestBase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [
            patch.object(db, "DB_PATH", self.root / "codeatlas.db"),
            patch.object(config, "WORKSPACES_DIR", self.root / "workspaces"),
        ]
        for item in self.patches:
            item.start()
        db.init_db()
        self.admin = {"username": "admin", "role": "admin", "id": 1}
        self.repos = {}
        for slug in ("storefront", "gateway", "warehouse"):
            repo = db.create_repo(
                slug, slug.title(), f"https://example.test/{slug}.git",
                "https", slug, status="published",
            )
            self.repos[slug] = repo
            graph = config.graph_path(slug)
            graph.parent.mkdir(parents=True)
            graph.write_text(json.dumps({"nodes": [], "links": []}))

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def make_group(self, slugs=("storefront", "gateway", "warehouse")):
        request = group_routes.CreateGroupRequest(
            slug="stack",
            name="Stack",
            description="storefront calls gateway",
            members=[group_routes.GroupMember(repo_slug=s) for s in slugs],
        )
        return group_routes.create_group(request, self.admin)["group"]


class GroupAdminTests(GroupTestBase):
    def test_create_list_update_delete(self):
        group = self.make_group()
        self.assertEqual([m["repo_slug"] for m in group["members"]],
                         ["storefront", "gateway", "warehouse"])
        self.assertEqual(len(group_routes.list_groups(self.admin)["groups"]), 1)

        updated = group_routes.update_group(
            "stack",
            group_routes.UpdateGroupRequest(
                name="Renamed",
                members=[
                    group_routes.GroupMember(repo_slug="gateway"),
                    group_routes.GroupMember(repo_slug="warehouse"),
                ],
            ),
            self.admin,
        )["group"]
        self.assertEqual(updated["name"], "Renamed")
        self.assertEqual(len(updated["members"]), 2)

        group_routes.delete_group("stack", self.admin)
        self.assertEqual(group_routes.list_groups(self.admin)["groups"], [])
        # Deleting a group never touches the repos themselves.
        self.assertIsNotNone(db.get_repo_by_slug("storefront"))

    def test_validation(self):
        with self.assertRaises(HTTPException) as too_few:
            self.make_group(("storefront",))
        self.assertEqual(too_few.exception.status_code, 400)
        with self.assertRaises(HTTPException) as unknown:
            self.make_group(("storefront", "missing"))
        self.assertEqual(unknown.exception.status_code, 404)
        with self.assertRaises(HTTPException) as duplicate:
            self.make_group(("storefront", "storefront"))
        self.assertEqual(duplicate.exception.status_code, 400)
        self.make_group()
        with self.assertRaises(HTTPException) as taken:
            self.make_group()
        self.assertEqual(taken.exception.status_code, 409)

    def test_deleting_a_repo_removes_only_its_membership(self):
        self.make_group()
        db.delete_repo("warehouse")
        members = db.get_repo_group_by_slug("stack")["members"]
        self.assertEqual([m["repo_slug"] for m in members], ["storefront", "gateway"])


class GroupAccessTests(GroupTestBase):
    def setUp(self):
        super().setUp()
        self.user = db.create_user("dev", "x", role="user")
        self.group = self.make_group()

    def test_user_needs_every_member_repo(self):
        user = {"id": self.user["id"], "role": "user"}
        self.assertFalse(db.user_can_access_repo_group(user, self.group))
        db.grant_access(user["id"], self.repos["storefront"]["id"])
        db.grant_access(user["id"], self.repos["gateway"]["id"])
        self.assertFalse(db.user_can_access_repo_group(user, self.group))
        with self.assertRaises(HTTPException) as denied:
            groups.require_group_for_user("stack", user)
        self.assertEqual(denied.exception.status_code, 403)
        self.assertEqual(group_routes.user_list_groups(user)["groups"], [])
        db.grant_access(user["id"], self.repos["warehouse"]["id"])
        self.assertTrue(db.user_can_access_repo_group(user, self.group))
        listed = group_routes.user_list_groups(user)["groups"]
        self.assertEqual([g["slug"] for g in listed], ["stack"])

    def test_admin_grant_gives_every_member_repo(self):
        group_routes.grant_group_access(
            "stack", group_routes.GroupGrantRequest(username="dev"), self.admin
        )
        user = {"id": self.user["id"], "role": "user"}
        self.assertTrue(db.user_can_access_repo_group(user, self.group))

    def test_admin_always_has_access_and_unknown_group_404s(self):
        self.assertTrue(db.user_can_access_repo_group({"role": "admin"}, self.group))
        with self.assertRaises(HTTPException) as missing:
            groups.require_group_for_user("nope", {"role": "admin"})
        self.assertEqual(missing.exception.status_code, 404)

    def test_resolve_requires_published_indexed_members(self):
        resolved = groups.resolve_group_members(self.group)
        self.assertEqual([t["slug"] for t in resolved], ["storefront", "gateway", "warehouse"])
        db.set_repo_status("warehouse", "indexed")
        with self.assertRaises(HTTPException) as unpublished:
            groups.resolve_group_members(db.get_repo_group_by_slug("stack"))
        self.assertEqual(unpublished.exception.status_code, 409)
        db.set_repo_status("warehouse", "published")
        config.graph_path("warehouse").unlink()
        with self.assertRaises(HTTPException) as unindexed:
            groups.resolve_group_members(db.get_repo_group_by_slug("stack"))
        self.assertEqual(unindexed.exception.status_code, 409)


class AdhocGroupTests(GroupTestBase):
    def setUp(self):
        super().setUp()
        self.user = db.create_user("dev", "x", role="user")
        self.dev = {"id": self.user["id"], "role": "user"}

    def test_user_can_pick_only_repos_they_hold(self):
        db.grant_access(self.dev["id"], self.repos["storefront"]["id"])
        db.grant_access(self.dev["id"], self.repos["gateway"]["id"])
        group = groups.adhoc_group_for_user("storefront,gateway", self.dev)
        self.assertEqual([m["repo_slug"] for m in group["members"]], ["gateway", "storefront"])
        self.assertEqual([t["slug"] for t in groups.resolve_group_members(group)],
                         ["gateway", "storefront"])
        for forbidden in ("storefront,warehouse", "storefront,nope"):
            with self.assertRaises(HTTPException) as denied:
                groups.adhoc_group_for_user(forbidden, self.dev)
            self.assertEqual(denied.exception.status_code, 403)
        with self.assertRaises(HTTPException) as missing:
            groups.adhoc_group_for_user("storefront,nope", self.admin)
        self.assertEqual(missing.exception.status_code, 404)

    def test_selection_size_and_identity(self):
        for bad in ("", "storefront", "storefront,storefront", "a,b,c,d,e,f,g"):
            with self.assertRaises(HTTPException) as invalid:
                groups.adhoc_group_for_user(bad, self.admin)
            self.assertEqual(invalid.exception.status_code, 400)
        first = groups.adhoc_group_for_user("storefront, gateway", self.admin)
        second = groups.adhoc_group_for_user("gateway,storefront,gateway", self.admin)
        other = groups.adhoc_group_for_user("storefront,warehouse", self.admin)
        self.assertEqual(first["slug"], second["slug"])
        self.assertNotEqual(first["slug"], other["slug"])
        self.assertTrue(first["slug"].startswith("adhoc-"))
        self.assertEqual(db.list_repo_groups(), [])

    def test_branch_can_be_pinned_per_repo(self):
        branch_ws = "storefront-dev"
        graph = config.graph_path(branch_ws)
        graph.parent.mkdir(parents=True)
        graph.write_text(json.dumps({"nodes": [], "links": []}))
        dev_branch = db.create_repo_branch(
            self.repos["storefront"]["id"], "dev", workspace=branch_ws, index_status="ready"
        )
        pending = db.create_repo_branch(
            self.repos["storefront"]["id"], "wip", workspace=None
        )
        foreign = db.create_repo_branch(
            self.repos["warehouse"]["id"], "other", workspace="warehouse-other", index_status="ready"
        )
        pinned = groups.adhoc_group_for_user(f"storefront@{dev_branch['id']},gateway", self.admin)
        default = groups.adhoc_group_for_user("storefront,gateway", self.admin)
        self.assertNotEqual(pinned["slug"], default["slug"])
        targets = {t["slug"]: t["workspace"] for t in groups.resolve_group_members(pinned)}
        self.assertEqual(targets["storefront"], branch_ws)
        self.assertEqual(targets["gateway"], "gateway")
        for bad, code in (
            (f"storefront@{foreign['id']},gateway", 404),
            ("storefront@999,gateway", 404),
            ("storefront@x,gateway", 400),
            (f"storefront@{dev_branch['id']},storefront,gateway", 400),
        ):
            with self.assertRaises(HTTPException) as invalid:
                groups.adhoc_group_for_user(bad, self.admin)
            self.assertEqual(invalid.exception.status_code, code, bad)
        not_ready = groups.adhoc_group_for_user(f"storefront@{pending['id']},gateway", self.admin)
        with self.assertRaises(HTTPException) as unindexed:
            groups.resolve_group_members(not_ready)
        self.assertEqual(unindexed.exception.status_code, 409)

    def test_branch_pin_does_not_bypass_repo_grants(self):
        branch = db.create_repo_branch(
            self.repos["warehouse"]["id"], "dev", workspace="warehouse", index_status="ready"
        )
        db.grant_access(self.dev["id"], self.repos["storefront"]["id"])
        db.grant_access(self.dev["id"], self.repos["gateway"]["id"])
        with self.assertRaises(HTTPException) as denied:
            groups.adhoc_group_for_user(f"storefront,warehouse@{branch['id']}", self.dev)
        self.assertEqual(denied.exception.status_code, 403)

    def test_unpublished_repo_is_rejected_when_resolving(self):
        db.set_repo_status("warehouse", "indexed")
        group = groups.adhoc_group_for_user("storefront,warehouse", self.admin)
        with self.assertRaises(HTTPException) as unpublished:
            groups.resolve_group_members(group)
        self.assertEqual(unpublished.exception.status_code, 409)


class GroupToolboxTests(unittest.TestCase):
    def test_calls_route_to_the_named_member_only(self):
        members = [
            {"slug": "storefront", "name": "Storefront", "workspace": "ws-storefront"},
            {"slug": "gateway", "name": "Gateway", "workspace": "ws-gateway"},
        ]
        with patch("app.agent.tools.RepositoryToolbox") as toolbox_class:
            toolbox_class.side_effect = lambda ws: SimpleNamespace(
                call=lambda name, args: json.dumps({"ok": True, "ws": ws})
            )
            toolbox_class._trace_summary = lambda result: {"ok": result.get("ok")}
            toolbox = GroupRepositoryToolbox(members)
            hit = json.loads(toolbox.call("search_code", {"repo": "gateway", "query": "x"}))
            self.assertEqual(hit["ws"], "ws-gateway")
            self.assertEqual(hit["repo_slug"], "gateway")
            by_name = json.loads(toolbox.call("search_code", {"repo": "Storefront"}))
            self.assertEqual(by_name["ws"], "ws-storefront")
            bad = json.loads(toolbox.call("search_code", {"repo": "warehouse"}))
            self.assertFalse(bad["ok"])
            missing = json.loads(toolbox.call("search_code", {}))
            self.assertFalse(missing["ok"])
        enum = toolbox.tool_definitions[0]["parameters"]["properties"]["repo"]["enum"]
        self.assertEqual(enum, ["storefront", "gateway"])


class GroupAskTests(GroupTestBase):
    def setUp(self):
        super().setUp()
        self.group = self.make_group()
        self.user = {"id": 7, "role": "admin", "username": "a", "user_type": "dev_team",
                     "_session_key": "s"}

    def request(self, **overrides):
        values = dict(
            question="how does a payment flow?", feedback_id=None, llm_mode="auto",
            conversation_id=None, follow_up=False, deep_investigation=False,
            answer_user_type=None, activity_request_id=None, user_llm=None,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def run_ask(self, request):
        captured = {}

        def fake_generate(context, **kwargs):
            captured["context"] = context
            captured["toolbox"] = kwargs["toolbox"]
            captured["question"] = kwargs["question"]
            captured["agent_context"] = kwargs["agent_context"]
            return {"answer": "answer", "provider_used": "shared",
                    "retrieval_mode": "agentic", "agent_trace": [], "tool_calls": 1}

        from app import main
        with patch.object(group_ask, "generate", fake_generate), patch.object(
            main, "build_context", return_value={"context_nodes": []}
        ), patch.object(main, "enforce_rate_limit"):
            return group_ask.answer_group_request(
                request, db.get_repo_group_by_slug("stack"), self.user
            ), captured

    def test_answer_covers_every_repo_and_threads_follow_ups(self):
        response, seen = self.run_ask(self.request())
        self.assertEqual(response["answer"], "answer")
        self.assertEqual(response["group"]["slug"], "stack")
        self.assertEqual([r["slug"] for r in response["group_repositories"]],
                         ["storefront", "gateway", "warehouse"])
        self.assertEqual(list(seen["toolbox"].repositories), ["storefront", "gateway", "warehouse"])
        self.assertIn("storefront calls gateway", seen["question"])
        self.assertEqual(seen["agent_context"], "")
        self.assertTrue(response["conversation_id"])

        follow, seen = self.run_ask(self.request(
            question="and on failure?", follow_up=True,
            conversation_id=response["conversation_id"],
        ))
        self.assertEqual(follow["conversation_id"], response["conversation_id"])
        self.assertIn("how does a payment flow?", seen["agent_context"])

    def test_repo_that_forbids_shared_llm_blocks_shared_tier_for_group(self):
        from app import main
        self.assertFalse(main.is_shared_llm_mode("personal"))
        seen = {}

        def fake_generate(context, **kwargs):
            seen["allow"] = kwargs["allow_shared_fallback"]
            return {"answer": "a", "provider_used": "p", "retrieval_mode": "agentic"}

        with patch.object(group_ask, "generate", fake_generate), patch.object(
            main, "build_context", return_value={}
        ), patch.object(main, "enforce_rate_limit"):
            group_ask.answer_group_request(
                self.request(), db.get_repo_group_by_slug("stack"), self.user
            )
        self.assertFalse(seen["allow"])  # repos default to allow_shared_fallback=0


if __name__ == "__main__":
    unittest.main()


class GroupRevisionTests(GroupTestBase):
    def test_editing_the_relationship_note_changes_the_cache_identity(self):
        from app import main
        group = self.make_group()
        targets = groups.resolve_group_members(group)
        before = group_ask.group_revision(main, targets, group)
        group_routes.update_group(
            "stack", group_routes.UpdateGroupRequest(description="new note"), self.admin
        )
        after = group_ask.group_revision(
            main, targets, db.get_repo_group_by_slug("stack")
        )
        self.assertNotEqual(before, after)
