import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app import db
from app.auth import routes as auth_routes


class UserChatPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(
            db,
            "DB_PATH",
            Path(self.temp.name) / "codeatlas.db",
        )
        self.db_patch.start()
        db.init_db()
        self.alice = db.create_user(
            "alice",
            "unused",
            email="alice@example.com",
        )
        self.bob = db.create_user(
            "bob",
            "unused",
            email="bob@example.com",
        )

    def tearDown(self):
        self.db_patch.stop()
        self.temp.cleanup()

    @staticmethod
    def chat_request(question="How does login work?", answer="Through auth routes."):
        return auth_routes.UserChatRequest(
            title=question,
            preview=answer,
            createdAt="2026-09-21T10:00:00.000Z",
            updatedAt="2026-09-21T10:01:00.000Z",
            askMode="single",
            workspace="codeatlas",
            branchId=1,
            llmMode="mimo",
            answerUserType="dev_team",
            turns=[
                auth_routes.ChatTurnPayload(
                    id="turn-1",
                    question=question,
                    answer=answer,
                    createdAt="2026-09-21T10:00:00.000Z",
                    answeredAt="2026-09-21T10:01:00.000Z",
                )
            ],
        )

    def test_chat_routes_are_scoped_to_the_authenticated_user(self):
        request = self.chat_request()
        saved = auth_routes.save_my_chat("chat-shared-id", request, self.alice)

        self.assertEqual(saved, {"saved": "chat-shared-id"})
        self.assertEqual(len(auth_routes.get_my_chats(self.alice)["chats"]), 1)
        self.assertEqual(auth_routes.get_my_chats(self.bob)["chats"], [])

        auth_routes.save_my_chat(
            "chat-shared-id",
            self.chat_request("What does Bob see?", "Only Bob's chat."),
            self.bob,
        )
        alice_chat = auth_routes.get_my_chats(self.alice)["chats"][0]
        bob_chat = auth_routes.get_my_chats(self.bob)["chats"][0]

        self.assertEqual(alice_chat["turns"][0]["question"], "How does login work?")
        self.assertEqual(bob_chat["turns"][0]["question"], "What does Bob see?")
        self.assertFalse(auth_routes.delete_my_chat("missing-chat", self.alice)["deleted"])
        self.assertTrue(auth_routes.delete_my_chat("chat-shared-id", self.alice)["deleted"])
        self.assertEqual(auth_routes.get_my_chats(self.alice)["chats"], [])
        self.assertEqual(len(auth_routes.get_my_chats(self.bob)["chats"]), 1)

    def test_upsert_updates_a_chat_without_duplicating_it(self):
        auth_routes.save_my_chat("chat-1", self.chat_request(), self.alice)
        auth_routes.save_my_chat(
            "chat-1",
            self.chat_request("Updated question", "Updated answer"),
            self.alice,
        )

        chats = db.list_user_chats(self.alice["id"])
        self.assertEqual(len(chats), 1)
        self.assertEqual(chats[0]["title"], "Updated question")
        self.assertEqual(chats[0]["turns"][0]["answer"], "Updated answer")

    def test_deleting_a_user_cascades_to_saved_chats(self):
        auth_routes.save_my_chat("chat-1", self.chat_request(), self.alice)

        db.delete_user(self.alice["id"])

        self.assertEqual(db.list_user_chats(self.alice["id"]), [])

    def test_schema_upgrade_preserves_existing_users_and_sessions(self):
        token = db.create_session(self.alice["id"])
        with db.connect() as connection:
            connection.execute("DROP TABLE user_chats")
            connection.execute("DROP TABLE answer_feedback")

        db.init_db()

        restored_user = db.get_user_by_email("alice@example.com")
        session_user = db.get_session_user(token)
        self.assertEqual(restored_user["id"], self.alice["id"])
        self.assertEqual(session_user["id"], self.alice["id"])
        self.assertEqual(db.list_user_chats(self.alice["id"]), [])
        self.assertEqual(db.list_answer_feedback(), [])

    def test_invalid_chat_id_and_payload_are_rejected(self):
        with self.assertRaises(HTTPException) as invalid_id:
            auth_routes.save_my_chat("not/a/chat", self.chat_request(), self.alice)
        self.assertEqual(invalid_id.exception.status_code, 400)

        oversized = self.chat_request()
        oversized.preview = "x" * 20_001
        with self.assertRaises(HTTPException) as invalid_payload:
            auth_routes.save_my_chat("chat-1", oversized, self.alice)
        self.assertEqual(invalid_payload.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
