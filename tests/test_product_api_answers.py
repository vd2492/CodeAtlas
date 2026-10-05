import unittest

from app.llm import client


class ProductDetailQuestionTests(unittest.TestCase):
    def test_api_questions_are_detected(self):
        for question in (
            "What is the API endpoint for placing an order?",
            "what all parameters are there in the request body of the create order api",
            "Show me the response body fields and their datatype",
            "what is the complete url for the rider login endpoint",
            "Which webhook does the payment gateway call?",
            "what status codes can the refund API return",
            "query params for the search call",
        ):
            self.assertTrue(client.is_product_api_question(question), question)

    def test_column_mapping_questions_are_detected(self):
        for question in (
            "what is the mapping of the order_status column?",
            "Which column stores the customer's pincode",
            "give me the field mapping between the rider app and the orders table",
            "what is the data type of delivery_slot",
            "list the database tables used for returns",
            "mapping of a column in the shipments table",
            "map the fields between the rider app and the orders table",
        ):
            self.assertTrue(client.is_product_data_question(question), question)

    def test_other_product_questions_are_unchanged(self):
        for question in (
            "I have to share the RTO logic to operations team, summarise it",
            "what happens after On Hold, IVR / customer confirmation logic",
            "how does login work for a rider?",
            "explain the rest of the flow",
            "what parameters decide the delivery slot?",
            "how does mapping of riders to hubs work?",
            "what happens in the two column layout on the hub screen",
            "which table view does the rider see for pending pickups",
            "",
        ):
            self.assertFalse(client.is_product_api_question(question), question)
            self.assertFalse(client.is_product_data_question(question), question)
            self.assertEqual(
                client.product_response_instruction(question),
                client.PRODUCT_TEAM_RESPONSE_INSTRUCTION,
            )
            self.assertEqual(
                client.product_query_suffix(question), client.PRODUCT_TEAM_QUERY_SUFFIX
            )

    def test_the_right_instruction_is_chosen(self):
        api = "what is the api endpoint and request body for creating an order"
        data = "what is the mapping of the order_status column"
        both = "which api fills the order_status column and what is its mapping"
        self.assertIs(
            client.product_response_instruction(api),
            client.PRODUCT_TEAM_API_RESPONSE_INSTRUCTION,
        )
        self.assertIs(
            client.product_response_instruction(data),
            client.PRODUCT_TEAM_DATA_RESPONSE_INSTRUCTION,
        )
        self.assertIs(
            client.product_response_instruction(both),
            client.PRODUCT_TEAM_API_DATA_RESPONSE_INSTRUCTION,
        )
        self.assertIn("complete endpoint URL", client.product_response_instruction(api))
        self.assertNotIn("DATA MAPPING", client.product_response_instruction(api))
        mapping = client.product_response_instruction(data)
        self.assertIn("table and column", mapping)
        self.assertIn("exactly which part", mapping)
        self.assertNotIn("API DETAILS", mapping)
        combined = client.product_response_instruction(both)
        self.assertIn("API DETAILS", combined)
        self.assertIn("DATA MAPPING", combined)
        for question in (api, data, both):
            self.assertIn("no file names", client.product_response_instruction(question))
            self.assertIn("table", client.product_query_suffix(data))

    def test_detail_questions_get_the_exception_everywhere_it_must(self):
        for question in (
            "what is the api endpoint for creating an order",
            "what is the mapping of the order_status column",
        ):
            instruction = client.product_response_instruction(question)

            class Toolbox:
                response_style_instruction = instruction
                tool_definitions = None

            self.assertIn(
                client.PRODUCT_DETAIL_PROMPT_NOTE, client._agent_system_prompt(Toolbox())
            )
            context = {"response_style_instruction": instruction}
            self.assertIn(client.PRODUCT_DETAIL_PROMPT_NOTE, client._system_prompt(context))
            self.assertEqual(client._product_toolbox_answer(Toolbox()), "detail")
            self.assertIn("columns and mappings", client._budget_exhausted_prompt("detail"))
            self.assertIn(
                "columns and mappings",
                client._agent_question("and the types?", "earlier evidence", "detail"),
            )

    def test_ordinary_product_prompts_are_untouched(self):
        plain = client.PRODUCT_TEAM_RESPONSE_INSTRUCTION

        class Toolbox:
            response_style_instruction = plain
            tool_definitions = None

        self.assertNotIn(
            client.PRODUCT_DETAIL_PROMPT_NOTE, client._agent_system_prompt(Toolbox())
        )
        self.assertNotIn(
            client.PRODUCT_DETAIL_PROMPT_NOTE,
            client._system_prompt({"response_style_instruction": plain}),
        )
        self.assertIs(client._product_toolbox_answer(Toolbox()), True)
        self.assertIn("APIs, endpoints", client._budget_exhausted_prompt(True))
        self.assertIn("APIs, endpoints", client._agent_question("q", "evidence", True))

    def test_product_cleanup_keeps_urls_but_still_strips_source_references(self):
        answer = (
            "Call POST https://api.example.com/v1/orders.json with `orderId`. "
            "See app/src/main/java/OrderService.kt:42-50 for details."
        )
        cleaned = client._clean_product_answer(answer, keep_urls=True)
        self.assertIn("https://api.example.com/v1/orders.json", cleaned)
        self.assertNotIn("OrderService.kt", cleaned)
        # Ordinary product answers are cleaned exactly as they always were.
        self.assertNotIn(
            "https://api.example.com/v1/orders.json", client._clean_product_answer(answer)
        )
        self.assertEqual(
            client._clean_product_answer("No links here, see src/app/Main.kt:3."),
            "No links here, see repository evidence.",
        )
        table = "| orders.order_status | varchar(32) | Not empty |"
        self.assertEqual(client._clean_product_answer(table), table)


if __name__ == "__main__":
    unittest.main()
