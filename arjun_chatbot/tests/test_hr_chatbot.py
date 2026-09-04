# Regression tests for arjun_chatbot/api/hr_chatbot.py
#
# The matching pipeline (regex -> fuzzy -> AI classify -> topic gate ->
# log/suggest -> help) has already caught several real false positives
# during manual testing (see the comments throughout hr_chatbot.py for
# each one). This file locks those specific cases in as automated tests
# so a future edit to the scoring/keyword logic can't silently
# reintroduce them.
#
# Routing tests never touch the DB or Groq: _dispatch is patched to
# report which intent it was asked to run instead of actually running a
# data-lookup handler (those need a real Employee record and are a
# separate concern from "did we pick the right intent"). AI fallback is
# explicitly disabled in setUp so no test makes a real network call.

from unittest.mock import patch

import frappe
from frappe.tests.utils import FrappeTestCase

from arjun_chatbot.api import hr_chatbot


def _route(message):
	"""What intent key (or sentinel) would `ask()` dispatch this message
	to, without actually running the handler? _dispatch is patched to
	report the entry it was given instead of calling entry["handler"] -
	keeps these tests fast, DB-free, and focused purely on routing."""
	captured = {}

	def fake_dispatch(entry, msg):
		captured["key"] = entry["key"]
		return {"reply": "__dispatched__"}

	with patch.object(hr_chatbot, "_dispatch", side_effect=fake_dispatch):
		reply = hr_chatbot.ask(message)

	if "key" in captured:
		return captured["key"]
	# Nothing matched cleanly enough to dispatch - inspect the actual
	# reply to tell apart off-topic / suggested / generic-help, since
	# those three all skip _dispatch entirely.
	text = reply["reply"]
	if text == hr_chatbot._off_topic():
		return "__off_topic__"
	if text.startswith("I'm not fully sure"):
		return "__suggested__"
	return None


class TestHRChatbotRouting(FrappeTestCase):
	"""Which intent (if any) a message resolves to - regex, fuzzy, and the
	interactions between them. Every case here reproduces a specific
	scenario already called out in hr_chatbot.py's own comments as a real
	bug found by testing, plus a spot-check of a few plain/typo'd phrases
	across the regex layer."""

	def setUp(self):
		# Deterministic: no test should depend on (or trigger) a real
		# Groq call. Restored automatically - FrappeTestCase runs each
		# test in a transaction that's rolled back afterwards.
		settings = frappe.get_single("HR Chatbot Settings")
		settings.enable_ai_fallback = 0
		settings.save()
		frappe.clear_cache(doctype="HR Chatbot Settings")

	# ---- regressions found by real manual testing (see hr_chatbot.py) ----

	def test_president_of_india_is_not_attendance(self):
		# "present" is a keyword unique to the attendance intent, so
		# "president" alone scores a full 1.0 by weight - but a 0.875
		# character-similarity ratio on a single matched word isn't close
		# enough to trust. Must not dispatch to attendance, and (this is
		# the part a naive topic-gate implementation gets wrong) must not
		# even look HR-related enough to suggest attendance either.
		self.assertEqual(_route("president of India"), "__off_topic__")

	def test_leave_alone_does_not_confidently_pick_leave_balance(self):
		# "leave" is shared by four intents, so on its own it can't win
		# outright - a bare "have i taken leave last month" needs a
		# second, more specific signal before committing to leave_balance.
		self.assertNotEqual(_route("have i taken leave last month"), "leave_balance")

	def test_blood_type_does_not_match_leave_types(self):
		# "type" was deliberately excluded from leave_types_policy's
		# keyword list because "what's my blood TYPE on file" fuzzy-
		# matched it via the single word "type" alone.
		self.assertEqual(_route("what's my blood TYPE on file"), "personal_details")

	def test_typo_tolerant_attendance(self):
		self.assertEqual(_route("atendance this month"), "attendance")

	def test_attendece_and_attendence_misspellings(self):
		# Real production case: "attendece on 29 June" scored only ~0.84
		# fuzzy similarity to "attendance" - below the single-word
		# confidence guard (0.90) - so it fell through to the generic
		# fallback instead of answering. Fixed by listing these specific,
		# real misspellings as their own keywords/regex alternatives
		# rather than loosening the guard (which would reopen the
		# president/present false positive it exists to block).
		self.assertEqual(_route("attendece on 29 June"), "attendance")
		self.assertEqual(_route("what is my attendence status today"), "attendance")

	def test_typo_tolerant_leave_balance(self):
		self.assertEqual(_route("how mnay leaves left"), "leave_balance")

	def test_salary_increase_question_does_not_return_payslip_data(self):
		# Real bug caught by live testing: "how can my salary be increase?"
		# (a career/policy question this app has no answer for) matched
		# the bare "my salary" pattern and confidently returned real
		# payslip data - wrong, not just unhelpful. Fixed at both layers:
		# the payslip regex's negative lookahead, and sharing the fuzzy
		# keyword "salary" with salary_breakup so a lone typo'd mention
		# no longer auto-resolves either. Genuine salary questions (with
		# or without a typo) must still work.
		self.assertNotEqual(_route("how can my salary be increase?"), "payslip")
		self.assertEqual(_route("what's my salary"), "payslip")
		self.assertEqual(_route("sallary slip please"), "payslip")

	def test_date_of_birth(self):
		self.assertEqual(_route("my date of birth"), "my_profile")
		self.assertEqual(_route("what's my dob"), "my_profile")

	def test_hr_contacts_routes_correctly_and_not_to_manager(self):
		# "who is the HR Manager" contains "manager" and must not be
		# caught by the reporting-manager intent, which is checked right
		# after this one and would otherwise win on that bare substring.
		self.assertEqual(_route("my hr name"), "hr_contacts")
		self.assertEqual(_route("who is the hr manager"), "hr_contacts")
		self.assertEqual(_route("who is my manager"), "manager")

	def test_company_info_routes_correctly(self):
		# Real live message: "Om Logistics owener?" - a typo of "owner",
		# with no other trigger phrase present. Must reach company_info,
		# not the off-topic redirect.
		self.assertEqual(_route("Om Logistics owener?"), "company_info")
		self.assertEqual(_route("who owns om logistics"), "company_info")
		self.assertEqual(_route("who founded the company"), "company_info")

	# ---- off-topic gate ----

	def test_off_topic_chit_chat_is_redirected(self):
		self.assertEqual(_route("what is the capital of France"), "__off_topic__")
		self.assertEqual(_route("what's the weather like today"), "__off_topic__")

	def test_off_topic_message_is_not_logged(self):
		before = frappe.db.count("HR Chatbot Unanswered Query")
		hr_chatbot.ask("what is the capital of France")
		after = frappe.db.count("HR Chatbot Unanswered Query")
		self.assertEqual(before, after)

	def test_hr_flavored_gap_is_not_off_topic(self):
		# Contains "company"/"staff" (broad HR vocabulary) but doesn't
		# match any actual intent's keywords - should NOT be redirected
		# as off-topic, should fall through to the gap log/help instead.
		self.assertNotEqual(
			_route("is there a gym membership discount for company staff"), "__off_topic__"
		)

	# ---- Hinglish (Latin-script transliterated Hindi) ----
	# See the scoping note above INTENTS in hr_chatbot.py for why this is
	# a small, deliberately narrow set rather than a general translation
	# layer.

	def test_hinglish_phrasings(self):
		cases = {
			"chutti kitni bachi hai": "leave_balance",
			"chutti kaise le": "apply_leave",
			"meri hazri kitni hai": "attendance",
			"meri tankhwah kitni hai": "salary_breakup",
			"meri tankhwah dikhao": "payslip",
			"istifa dena hai": "resignation",
			"kharcha claim karna hai": "expense_claim",
			"bank khata number kya hai": "bank_details",
		}
		for message, expected_key in cases.items():
			with self.subTest(message=message):
				self.assertEqual(_route(message), expected_key)

	def test_namaste_and_dhanyawad_are_greeting_and_thanks(self):
		# Greeting/thanks have key=None - the assertion is just that they
		# dispatch cleanly (not None-vs-something-else routing confusion,
		# not an unmatched fallthrough to off-topic/help).
		captured = {}

		def fake_dispatch(entry, msg):
			captured["called"] = True
			return {"reply": "x"}

		with patch.object(hr_chatbot, "_dispatch", side_effect=fake_dispatch):
			hr_chatbot.ask("namaste")
		self.assertTrue(captured.get("called"))

		captured.clear()
		with patch.object(hr_chatbot, "_dispatch", side_effect=fake_dispatch):
			hr_chatbot.ask("dhanyawad")
		self.assertTrue(captured.get("called"))

	# ---- regex layer spot-checks (a sample across the 28 intents, not
	# exhaustive - this is a smoke test that the dispatch table itself
	# still wires up correctly, not a full behavioral spec) ----

	def test_regex_layer_samples(self):
		cases = {
			"what's my leave balance": "leave_balance",
			"how do I apply for leave": "apply_leave",
			"status of my leave request": "leave_status",
			"my payslip for march": "payslip",
			"what's my ctc": "salary_breakup",
			"which bank account is my salary going to": "bank_details",
			"who is my manager": "manager",
			"what's my notice period": "notice_period",
			"my pan number": "statutory_ids",
			"when's the next holiday": "holiday",
		}
		for message, expected_key in cases.items():
			with self.subTest(message=message):
				self.assertEqual(_route(message), expected_key)

	def test_greeting_and_thanks_do_not_dispatch_a_data_intent(self):
		# Greeting/thanks entries have key=None (see _entry(None, ...)) -
		# they still go through _dispatch, so _route reports None here,
		# same as "nothing matched". The real assertion is that they
		# don't accidentally win as some unrelated keyed intent.
		self.assertIsNone(_route("hi"))
		self.assertIsNone(_route("thanks!"))


class TestFuzzyIsConfident(FrappeTestCase):
	"""Direct unit tests on the shared trust-check helper, independent of
	ask()'s routing - see its docstring for why the single-word guard
	exists at all."""

	def test_rejects_below_threshold(self):
		self.assertFalse(hr_chatbot._fuzzy_is_confident(object(), 0.4, 2, 1.0, threshold=0.5))

	def test_rejects_none_entry(self):
		self.assertFalse(hr_chatbot._fuzzy_is_confident(None, 5.0, 1, 1.0, threshold=0.5))

	def test_rejects_weak_single_word_match_even_above_threshold(self):
		# The exact "president"/"present" shape: full score, but resting
		# on one weakly-matched word.
		self.assertFalse(
			hr_chatbot._fuzzy_is_confident(object(), 1.0, 1, 0.875, threshold=1.0)
		)

	def test_accepts_strong_single_word_match(self):
		self.assertTrue(
			hr_chatbot._fuzzy_is_confident(object(), 1.0, 1, 0.95, threshold=1.0)
		)

	def test_accepts_multi_word_match_even_if_individually_weak(self):
		# Two independent words both pointing the same way is trusted
		# even if neither alone would clear the single-word bar.
		self.assertTrue(
			hr_chatbot._fuzzy_is_confident(object(), 1.0, 2, 0.8, threshold=1.0)
		)


class TestUnansweredQueryLogging(FrappeTestCase):
	def setUp(self):
		settings = frappe.get_single("HR Chatbot Settings")
		settings.enable_ai_fallback = 0
		settings.save()
		frappe.clear_cache(doctype="HR Chatbot Settings")

	def test_genuine_gap_is_logged_with_diagnostics(self):
		before = frappe.db.count("HR Chatbot Unanswered Query")
		hr_chatbot.ask("is there a gym membership discount for company staff")
		after = frappe.db.count("HR Chatbot Unanswered Query")
		self.assertEqual(after, before + 1)

		row = frappe.get_last_doc("HR Chatbot Unanswered Query")
		self.assertEqual(row.question, "is there a gym membership discount for company staff")
		self.assertEqual(row.user, "Administrator")
		self.assertEqual(row.ai_fallback_attempted, 0)


class TestFollowUpMemory(FrappeTestCase):
	"""_remember_intent/_recall_intent and the bare-period follow-up
	branch in ask(). Cache is Redis-backed, not DB-backed, so
	FrappeTestCase's transaction rollback doesn't clean it up - each test
	clears its own key explicitly."""

	def setUp(self):
		settings = frappe.get_single("HR Chatbot Settings")
		settings.enable_ai_fallback = 0
		settings.save()
		frappe.clear_cache(doctype="HR Chatbot Settings")
		frappe.cache().delete_value(hr_chatbot._last_intent_cache_key())

	def tearDown(self):
		frappe.cache().delete_value(hr_chatbot._last_intent_cache_key())

	def test_recall_is_none_with_nothing_remembered(self):
		self.assertIsNone(hr_chatbot._recall_intent())

	def test_remember_then_recall_round_trip(self):
		# Regression for a real bug: frappe.cache().get_value() caches a
		# "not found" None locally in-process unless expires=True is
		# passed for a key written with expires_in_sec - calling
		# _recall_intent() (or anything hitting the same key) BEFORE the
		# value exists must not poison later calls after it's set. See
		# _recall_intent's docstring.
		self.assertIsNone(hr_chatbot._recall_intent())
		hr_chatbot._remember_intent("attendance")
		entry = hr_chatbot._recall_intent()
		self.assertIsNotNone(entry)
		self.assertEqual(entry["key"], "attendance")

	def test_only_followup_capable_intents_are_remembered(self):
		hr_chatbot._remember_intent("manager")  # not in _FOLLOWUP_CAPABLE_INTENTS
		self.assertIsNone(hr_chatbot._recall_intent())

	def test_dispatch_remembers_followup_capable_intent(self):
		with patch.object(hr_chatbot, "_current_employee", return_value=None):
			hr_chatbot.ask("my attendance this month")
		entry = hr_chatbot._recall_intent()
		self.assertIsNotNone(entry)
		self.assertEqual(entry["key"], "attendance")

	def test_bare_period_followup_routes_to_remembered_intent(self):
		# entry["handler"] is a direct function reference captured once
		# when INTENTS was built, not a late lookup by name - patching
		# hr_chatbot._attendance itself wouldn't be seen by _dispatch, so
		# the stub is swapped into the INTENTS_BY_KEY entry (the same
		# dict object INTENTS holds) instead.
		from unittest.mock import MagicMock

		stub = MagicMock(return_value="ATTENDANCE ANSWER")
		with patch.object(hr_chatbot, "_current_employee", return_value="some-employee"):
			with patch.dict(hr_chatbot.INTENTS_BY_KEY["attendance"], {"handler": stub}):
				hr_chatbot.ask("my attendance this month")
				reply = hr_chatbot.ask("what about last month")

		self.assertEqual(reply["reply"], "ATTENDANCE ANSWER")
		# Called twice: once for the original message, once for the bare
		# follow-up - and the follow-up's raw message text (not the
		# original) is what gets passed through, so the handler's own
		# _extract_period(message) call re-parses "last month" itself.
		self.assertEqual(stub.call_count, 2)
		self.assertEqual(stub.call_args_list[1].args[1], "what about last month")

	def test_cold_followup_with_nothing_remembered_is_not_hijacked(self):
		# No prior dispatch in this test - "what about last month" names
		# a period but has no HR keywords of its own, so with nothing
		# remembered it must fall through to the off-topic gate, not
		# error or hang on a missing intent.
		reply = hr_chatbot.ask("what about last month")
		self.assertEqual(reply["reply"], hr_chatbot._off_topic())

	def test_unrelated_new_question_is_not_hijacked_by_memory(self):
		# A message that matches its own intent must win outright,
		# regardless of what's remembered from the previous turn.
		with patch.object(hr_chatbot, "_current_employee", return_value="some-employee"):
			with patch.dict(hr_chatbot.INTENTS_BY_KEY["attendance"], {"handler": lambda *a: "ATTENDANCE ANSWER"}):
				hr_chatbot.ask("my attendance this month")
			with patch.dict(hr_chatbot.INTENTS_BY_KEY["manager"], {"handler": lambda *a: "MANAGER ANSWER"}):
				reply = hr_chatbot.ask("who is my manager")
		self.assertEqual(reply["reply"], "MANAGER ANSWER")

	def test_period_naming_question_about_a_different_topic_is_not_hijacked(self):
		# Real bug caught by live testing: "on which date I take leave in
		# august" was hijacked into an attendance answer purely because
		# "attendance" was the remembered intent and "august" is a period
		# - even though the message plainly says "leave" and never
		# mentions attendance. A message naming a period is only deferred
		# to memory when it has NO competing signal of its own; here
		# "leave" (however weak/unconfident alone) points at a different
		# intent than the remembered one, so memory must be refused.
		with patch.object(hr_chatbot, "_current_employee", return_value="some-employee"):
			with patch.dict(hr_chatbot.INTENTS_BY_KEY["attendance"], {"handler": lambda *a: "ATTENDANCE ANSWER"}):
				hr_chatbot.ask("my attendance this month")
			reply = hr_chatbot.ask("on which date I take leave in august")
		self.assertNotEqual(reply["reply"], "ATTENDANCE ANSWER")
