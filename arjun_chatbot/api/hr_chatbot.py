# arjun_chatbot/api/hr_chatbot.py
#
# A query-resolver for the HRMS desk. Three layers, tried in order, cheapest
# and most exact first:
#   1. exact keyword/regex intents;
#   2. typo-tolerant fuzzy matching (difflib, still fully local/free);
#   3. an OPTIONAL AI fallback (Groq's free API - see HR Chatbot Settings),
#      used only when 1 and 2 both come up empty, and only to classify
#      *which* of the same fixed intents below the message maps to. It
#      never sees any employee/HR data and never writes the final answer
#      itself - it gets nothing but the question text, and returns nothing
#      but an intent name. The actual lookup and reply are always written
#      by this file's own code, same as layers 1 and 2. Off by default;
#      nothing is sent anywhere unless HR Chatbot Settings has it enabled
#      with an API key configured.
#
# Two kinds of answers, whichever layer picked the intent:
#   - data lookups, scoped to the logged-in user's own linked Employee
#     record (leave balance, leave status, attendance, payslip, holidays,
#     reporting manager, notice period) - via the same permission-checked
#     calls the rest of the app already uses (e.g. hrms's own
#     get_leave_details);
#   - fixed how-to answers for common "how do I..." questions about using
#     this HRMS (applying for leave, attendance regularization, comp off,
#     expense claims) and leave-type/policy info pulled straight from the
#     real Leave Type records - these don't need an Employee record, so
#     they work for anyone logged in.
# Unrecognised questions get a fixed "here's what I can answer" list
# rather than a guessed/hallucinated reply.

import calendar
import json
import re
from difflib import SequenceMatcher, get_close_matches

import frappe
from frappe import _
from frappe.utils import (
	add_days,
	add_months,
	fmt_money,
	formatdate,
	get_first_day,
	get_last_day,
	get_year_start,
	getdate,
	nowdate,
)
from frappe.utils.password import get_decrypted_password

# Words shorter than this are too generic on their own to fuzzy-match
# reliably ("my", "is", "of"...) - skip them in the typo-tolerant fallback.
_FUZZY_MIN_WORD_LEN = 4
_FUZZY_CUTOFF = 0.75
# Minimum weighted score (see _keyword_intent_count) to accept a fuzzy
# match at all - e.g. one distinctive keyword (weight 1.0), or two words
# each shared by only two intents (weight 0.5 + 0.5). Below this, the
# message is too ambiguous for local matching to guess confidently.
_FUZZY_ACCEPT_THRESHOLD = 1.0
# A match resting on just one fuzzy-matched word needs to be this close to
# count at all - "president" vs "present" is 0.875, well past the normal
# 0.75 cutoff, and was a real false positive (see _fuzzy_match).
_FUZZY_SINGLE_MATCH_CUTOFF = 0.90

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_TIMEOUT = 6


def _current_employee():
	"""The active Employee record linked to the logged-in user, or None.
	Guest can never have one (this API isn't reachable while logged out
	anyway, desk requires a session) - Administrator isn't special-cased:
	if someone deliberately links an Employee to it for testing, honour
	that link rather than hardcoding a refusal."""
	if frappe.session.user == "Guest":
		return None
	return frappe.db.get_value("Employee", {"user_id": frappe.session.user, "status": "Active"}, "name")


def _dispatch(entry, message):
	handler, needs_employee = entry["handler"], entry["needs_employee"]
	if not needs_employee:
		return {"reply": handler()}

	employee = _current_employee()
	if not employee:
		return {
			"reply": _(
				"I couldn't find an Employee record linked to your account, so I "
				"can't look that up for you. Please check with HR."
			)
		}
	try:
		return {"reply": handler(employee, message)}
	except Exception:
		frappe.log_error(title="HR chatbot handler failed")
		return {"reply": _("Something went wrong looking that up. Please try again or ask HR directly.")}


_keyword_intent_count_cache = None


def _keyword_intent_count():
	"""How many different intents share each keyword - "leave" appears in
	four intents' lists, "payslip" in only one. Used to down-weight generic
	shared words in fuzzy scoring below, computed once and cached (built
	lazily, not at import time, so it doesn't care about definition order
	relative to INTENTS)."""
	global _keyword_intent_count_cache
	if _keyword_intent_count_cache is None:
		counts = {}
		for entry in INTENTS:
			for kw in entry["keywords"]:
				counts[kw] = counts.get(kw, 0) + 1
		_keyword_intent_count_cache = counts
	return _keyword_intent_count_cache


def _fuzzy_match(message):
	"""Typo-tolerant fallback for when no regex intent matched cleanly -
	e.g. "how mnay leaves left" (typo for "many"). Scores each intent by
	its keyword matches, weighted DOWN for words shared across many
	intents - a bare "leave" alone is too weak a signal to confidently
	pick between leave_balance/apply_leave/leave_status/leave_types_policy,
	so it can't win on its own (caught by real testing: "have i taken
	leave last month" was wrongly resolving to leave_balance just because
	both share the word "leave"). Requires a minimum combined score to
	accept a match at all - weak/ambiguous messages fall through to the AI
	layer (or the help text) instead of confidently guessing wrong.

	A single matching word is extra risky: "president" vs the keyword
	"present" scores 0.875 - well past the normal cutoff, and a real bug
	caught by testing ("president of India" was answered with attendance
	data). Genuine unrelated words can coincidentally look a lot like a
	keyword; two independent words both pointing the same way is much
	stronger corroboration than one. So a single-word match must be a much
	closer, near-exact-typo ratio to be trusted at all - see
	_FUZZY_SINGLE_MATCH_CUTOFF.

	Ties broken by INTENTS order, so more specific intents listed earlier
	still win. No AI involved - this is plain edit-distance matching
	against a fixed vocabulary."""
	words = [w for w in re.findall(r"[a-zA-Z]+", message.lower()) if len(w) >= _FUZZY_MIN_WORD_LEN]
	if not words:
		return None

	keyword_counts = _keyword_intent_count()
	best_entry, best_score, best_word_count, best_min_ratio = None, 0.0, 0, 0.0
	for entry in INTENTS:
		if not entry["keywords"]:
			continue
		score, matched_words, min_ratio = 0.0, 0, 1.0
		for w in words:
			match = get_close_matches(w, entry["keywords"], n=1, cutoff=_FUZZY_CUTOFF)
			if match:
				score += 1.0 / keyword_counts[match[0]]
				matched_words += 1
				min_ratio = min(min_ratio, SequenceMatcher(None, w, match[0]).ratio())
		if score > best_score:
			best_entry, best_score, best_word_count, best_min_ratio = entry, score, matched_words, min_ratio

	if best_entry is None or best_score < _FUZZY_ACCEPT_THRESHOLD:
		return None
	if best_word_count == 1 and best_min_ratio < _FUZZY_SINGLE_MATCH_CUTOFF:
		return None
	return best_entry


def _ai_classify(message):
	"""Last-resort fallback: ask Groq's free API which known intent this
	message maps to, if HR Chatbot Settings has it enabled and configured.
	Sends ONLY the raw question text and the fixed list of intent names -
	no employee data, no DB content, nothing else. Returns an INTENTS entry
	or None (missing config, disabled, network error, or the model
	confidently said none of them fit - never raises, never blocks the
	reply)."""
	settings = frappe.get_cached_doc("HR Chatbot Settings")
	if not settings.get("enable_ai_fallback"):
		return None

	api_key = get_decrypted_password("HR Chatbot Settings", "HR Chatbot Settings", "groq_api_key", raise_exception=False)
	if not api_key:
		return None

	valid_keys = list(INTENTS_BY_KEY.keys())
	system_prompt = (
		"You classify a short HR-system question into exactly one of these "
		"intent names, or null if none fit: " + ", ".join(valid_keys) + ". "
		"Reply with ONLY a JSON object like {\"intent\": \"leave_balance\"} or "
		"{\"intent\": null}. No other text."
	)

	try:
		import requests

		response = requests.post(
			_GROQ_URL,
			headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
			json={
				"model": settings.get("groq_model") or "openai/gpt-oss-20b",
				"messages": [
					{"role": "system", "content": system_prompt},
					{"role": "user", "content": message},
				],
				"temperature": 0,
				# gpt-oss models "think" in a separate reasoning field before
				# the actual answer - low effort keeps that short, and 200
				# tokens leaves enough room for both the reasoning and the
				# final one-line JSON reply (30 was cutting content off
				# entirely, discovered while testing this against Groq for
				# real).
				"reasoning_effort": "low",
				"max_tokens": 200,
			},
			timeout=_GROQ_TIMEOUT,
		)
		response.raise_for_status()
		content = response.json()["choices"][0]["message"]["content"].strip()
		content = re.sub(r"^```(json)?|```$", "", content, flags=re.I).strip()
		intent_key = json.loads(content).get("intent")
	except Exception:
		frappe.log_error(title="HR chatbot AI fallback failed")
		return None

	return INTENTS_BY_KEY.get(intent_key)


@frappe.whitelist()
def ask(message: str) -> dict:
	message = (message or "").strip()
	if not message:
		return {"reply": _help()}

	for entry in INTENTS:
		if entry["pattern"].search(message):
			return _dispatch(entry, message)

	fuzzy_entry = _fuzzy_match(message)
	if fuzzy_entry:
		return _dispatch(fuzzy_entry, message)

	ai_entry = _ai_classify(message)
	if ai_entry:
		return _dispatch(ai_entry, message)

	return {"reply": _help()}


# ---- data lookups (need a linked Employee record) ----
#
# This covers essentially every field on Employee an employee would
# reasonably ask about their own record - profile, contact/emergency
# details, approvers, shift, bank, statutory IDs, salary breakup,
# education/work history, passport, personal/health details. One field
# is deliberately left out: custom_remarks (HR-internal notes about the
# employee) - that's HR's own working note, not something to hand back
# through self-service chat. Exit-only fields (relieving date, reason
# for leaving, exit feedback...) are also left out for now since they
# only apply to employees who've already left, not the active-employee
# case this is built for.

_MONTH_NAMES = {
	"jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
	"apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
	"aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
	"october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def _extract_period(message):
	"""Best-effort month extraction from free text - "this month"/"last
	month" resolve outright to (month, year); a bare month name ("feb",
	"february") returns (month, None) - deliberately NOT guessing which
	year, since that risks confidently answering "no data" when a record
	for a different year actually exists. Callers with a real employee to
	check against resolve the year from the DB instead (the most recent
	matching record) - see _resolve_year(). Returns None if the message
	doesn't name a period at all."""
	msg = (message or "").lower()

	if re.search(r"\bthis month\b|\bcurrent month\b", msg):
		d = getdate(nowdate())
		return d.month, d.year

	if re.search(r"\blast month\b|\bprevious month\b", msg):
		d = getdate(add_months(nowdate(), -1))
		return d.month, d.year

	match = re.search(r"\b(" + "|".join(_MONTH_NAMES.keys()) + r")\b\.?\s*('?\d{2,4})?", msg)
	if match:
		month = _MONTH_NAMES[match.group(1)]
		year_str = match.group(2)
		year = None
		if year_str:
			year = int(year_str.lstrip("'"))
			if year < 100:
				year += 2000
		return month, year

	return None


def _extract_day(message):
	"""Same idea as _extract_period but for one specific day - "yesterday",
	"today", or a day + month in either order ("5th july", "on 5 july",
	"july 5th"), optional year. Returns (day, month, year_or_None) or None.
	year is left unresolved for the same reason as _extract_period."""
	msg = (message or "").lower()
	# Typo-tolerant, same reasoning as _fuzzy_match - "yesterdat" (a real
	# typo caught in testing) shouldn't silently fall through to a whole-
	# month summary just because it doesn't match \byesterday\b exactly.
	words = re.findall(r"[a-zA-Z]+", msg)
	if any(get_close_matches(w, ["yesterday"], n=1, cutoff=_FUZZY_CUTOFF) for w in words):
		d = getdate(add_days(nowdate(), -1))
		return d.day, d.month, d.year
	if any(get_close_matches(w, ["today"], n=1, cutoff=_FUZZY_CUTOFF) for w in words):
		d = getdate(nowdate())
		return d.day, d.month, d.year

	# The ordinal suffix is matched loosely ([a-z]{0,2}, not literally
	# st/nd/rd/th) - "5ht august" (a real typo caught in testing, "ht"
	# instead of "th") otherwise fails to match at all and silently falls
	# through to the whole-month summary instead of the one day asked for.
	months = "|".join(_MONTH_NAMES.keys())
	match = re.search(r"\b(\d{1,2})[a-z]{0,2}\s*(?:of\s+)?(" + months + r")\b\.?\s*('?\d{2,4})?", msg)
	if match:
		day, month, year_str = int(match.group(1)), _MONTH_NAMES[match.group(2)], match.group(3)
	else:
		match = re.search(r"\b(" + months + r")\b\.?\s*(\d{1,2})[a-z]{0,2}\s*('?\d{2,4})?", msg)
		if not match:
			return None
		month, day, year_str = _MONTH_NAMES[match.group(1)], int(match.group(2)), match.group(3)

	year = None
	if year_str:
		year = int(year_str.lstrip("'"))
		if year < 100:
			year += 2000

	return day, month, year


def _resolve_year(doctype, date_field, employee, month):
	"""When a month was named without a year, find the actual year from
	real data instead of guessing: the most recent record for this
	employee whose date falls in that month, across all years. Falls back
	to the current year only if nothing at all matches (so a genuine
	"no data" reply is still checking a sensible year, not an arbitrary
	one)."""
	row = frappe.db.sql(
		f"""select {date_field} as d from `tab{doctype}`
		where employee=%(employee)s and docstatus=1 and MONTH({date_field})=%(month)s
		order by {date_field} desc limit 1""",
		{"employee": employee, "month": month},
		as_dict=True,
	)
	return getdate(row[0].d).year if row else getdate(nowdate()).year


def _leave_balance(employee, message=None):
	# "how many leaves have I taken" matches this same intent's keywords
	# ("many", "leave") but is asking the opposite question from "how many
	# leaves do I have left" - caught by real testing, where it was
	# wrongly answered with the remaining balance instead of what was
	# actually taken. Both answers come from the same underlying data
	# (hrms's get_leave_details already computes leaves_taken alongside
	# remaining_leaves), so this stays one intent - it just reads the
	# question to decide which figure to surface.
	if re.search(r"\btaken\b|\bused\b|\bconsumed\b|\bavailed\b", (message or "").lower()):
		return _leave_taken(employee, message)

	from hrms.hr.doctype.leave_application.leave_application import get_leave_details

	allocation = (get_leave_details(employee, nowdate()) or {}).get("leave_allocation") or {}
	if not allocation:
		return _("You don't have any leave allocated right now.")

	lines = [
		_("{0}: {1} remaining (of {2})").format(leave_type, data.get("remaining_leaves"), data.get("total_leaves"))
		for leave_type, data in allocation.items()
	]
	return _("Your leave balance:") + "<br>" + "<br>".join(lines)


def _leave_taken(employee, message=None):
	period = _extract_period(message or "")
	if period:
		month, year = period
		year = year or _resolve_year("Leave Application", "from_date", employee, month)
		start = getdate(f"{year}-{month:02d}-01")
		end = get_last_day(start)
		label = formatdate(start, "MMMM yyyy")
	else:
		# No period named - default to the current leave year to date,
		# not "ever", so this stays a manageable-sized real answer.
		start, end, label = get_year_start(nowdate()), nowdate(), _("this year")

	rows = frappe.get_all(
		"Leave Application",
		filters={
			"employee": employee,
			"status": "Approved",
			"docstatus": 1,
			"from_date": ["<=", end],
			"to_date": [">=", start],
		},
		fields=["leave_type", "total_leave_days"],
	)
	if not rows:
		return _("You haven't taken any leave in {0}.").format(label)

	totals = {}
	for r in rows:
		totals[r.leave_type] = totals.get(r.leave_type, 0) + (r.total_leave_days or 0)
	lines = [_("{0}: {1} day(s)").format(leave_type, days) for leave_type, days in totals.items()]
	return _("Leave taken in {0}:").format(label) + "<br>" + "<br>".join(lines)


def _leave_status(employee, message=None):
	rows = frappe.get_all(
		"Leave Application",
		filters={"employee": employee},
		fields=["leave_type", "from_date", "to_date", "status"],
		order_by="creation desc",
		limit=3,
	)
	if not rows:
		return _("You haven't applied for any leave yet.")

	lines = [
		_("{0}: {1} to {2} - {3}").format(r.leave_type, formatdate(r.from_date), formatdate(r.to_date), r.status)
		for r in rows
	]
	return _("Your recent leave requests:") + "<br>" + "<br>".join(lines)


def _attendance(employee, message=None):
	day_info = _extract_day(message or "")
	if day_info:
		day, month, year = day_info
		if year:
			try:
				target = getdate(f"{year}-{month:02d}-{day:02d}")
			except ValueError:
				return _("That doesn't look like a valid date.")
			status = frappe.db.get_value(
				"Attendance", {"employee": employee, "attendance_date": target, "docstatus": 1}, "status"
			)
			label = formatdate(target, "d MMMM yyyy")
		else:
			# No year named - read it straight from real data (the most
			# recent day/month match on record), rather than guessing.
			found = frappe.db.sql(
				"""select attendance_date, status from `tabAttendance`
				where employee=%(employee)s and docstatus=1
				and DAY(attendance_date)=%(day)s and MONTH(attendance_date)=%(month)s
				order by attendance_date desc limit 1""",
				{"employee": employee, "day": day, "month": month},
				as_dict=True,
			)
			if found:
				status, label = found[0].status, formatdate(found[0].attendance_date, "d MMMM yyyy")
			else:
				status, label = None, "{0} {1}".format(day, calendar.month_name[month])

		if not status:
			return _("No attendance was marked for you on {0}.").format(label)
		return _("You were marked {0} on {1}.").format(status, label)

	period = _extract_period(message or "")
	if period:
		month, year = period
		year = year or _resolve_year("Attendance", "attendance_date", employee, month)
		start = getdate(f"{year}-{month:02d}-01")
		end = get_last_day(start)
		label = formatdate(start, "MMMM yyyy")
	else:
		start, end = get_first_day(nowdate()), get_last_day(nowdate())
		label = formatdate(start, "MMMM yyyy")

	rows = frappe.get_all(
		"Attendance",
		filters={"employee": employee, "attendance_date": ["between", [start, end]], "docstatus": 1},
		fields=["status"],
	)
	if not rows:
		return _("No attendance was marked for you in {0}.").format(label)

	counts = {}
	for r in rows:
		counts[r.status] = counts.get(r.status, 0) + 1
	lines = [_("{0}: {1} day(s)").format(status, count) for status, count in counts.items()]
	return _("Your attendance in {0}:").format(label) + "<br>" + "<br>".join(lines)


def _payslip(employee, message=None):
	period = _extract_period(message or "")

	filters = {"employee": employee, "docstatus": 1}
	label = None
	if period:
		month, year = period
		year = year or _resolve_year("Salary Slip", "end_date", employee, month)
		start = getdate(f"{year}-{month:02d}-01")
		end = get_last_day(start)
		label = formatdate(start, "MMMM yyyy")
		# Overlap check, not equality - a slip's period might not align
		# exactly to calendar-month boundaries on every structure.
		filters["start_date"] = ["<=", end]
		filters["end_date"] = [">=", start]

	rows = frappe.get_all(
		"Salary Slip",
		filters=filters,
		fields=["name", "start_date", "end_date", "net_pay", "status"],
		order_by="end_date desc",
		limit=1,
	)
	if not rows:
		if period:
			return _("No payslip found for {0}.").format(label)
		return _("No payslip has been generated for you yet.")

	r = rows[0]
	if period:
		return _("Your payslip for {0} ({1} to {2}): net pay {3}, status {4}. <a href='/app/salary-slip/{5}'>View it here</a>.").format(
			label, formatdate(r.start_date), formatdate(r.end_date), fmt_money(r.net_pay), r.status, r.name
		)
	return _("Your latest payslip ({0} to {1}): net pay {2}, status {3}. <a href='/app/salary-slip/{4}'>View it here</a>.").format(
		formatdate(r.start_date), formatdate(r.end_date), fmt_money(r.net_pay), r.status, r.name
	)


def _next_holiday(employee, message=None):
	holiday_list = frappe.db.get_value("Employee", employee, "holiday_list")
	if not holiday_list:
		company = frappe.db.get_value("Employee", employee, "company")
		holiday_list = company and frappe.db.get_value("Company", company, "default_holiday_list")
	if not holiday_list:
		return _("No holiday list is set up for you.")

	rows = frappe.get_all(
		"Holiday",
		filters={"parent": holiday_list, "holiday_date": [">=", nowdate()]},
		fields=["holiday_date", "description"],
		order_by="holiday_date asc",
		limit=1,
	)
	if not rows:
		return _("No upcoming holidays found on your holiday list.")

	r = rows[0]
	return _("Your next holiday is {0} on {1}.").format(r.description or _("Holiday"), formatdate(r.holiday_date))


def _manager(employee, message=None):
	reports_to = frappe.db.get_value("Employee", employee, "reports_to")
	if not reports_to:
		return _("You don't have a reporting manager set in the system.")
	manager_name = frappe.db.get_value("Employee", reports_to, "employee_name")
	return _("Your reporting manager is {0}.").format(manager_name)


def _notice_period(employee, message=None):
	days = frappe.db.get_value("Employee", employee, "notice_number_of_days")
	if not days:
		return _("No notice period is set on your employee record. Please check with HR.")
	return _("Your notice period is {0} day(s).").format(days)


def _resignation(employee, message=None):
	# Real gap caught by testing: "I want to exit the company" had no
	# matching intent at all, so it got mapped to the closest EXISTING
	# concept instead (notice_period, then apply_leave) - wrong both times.
	# Employee Separation in this system is System Manager only (checked
	# the doctype's own permissions), so there's genuinely no self-service
	# form to point to here, unlike leave/attendance/expense claims - the
	# honest answer is "talk to HR", not a fabricated search-bar shortcut.
	days = frappe.db.get_value("Employee", employee, "notice_number_of_days")
	notice_line = _(" Your notice period is {0} day(s).").format(days) if days else ""
	return _(
		"Resigning isn't a self-service form in this system - there's "
		"nothing you can submit yourself for it. Please reach out to HR or "
		"your reporting manager directly to start the process.{0}"
	).format(notice_line)


def _mask_tail(value, keep=4):
	"""Show only the last few characters of an ID/account number. Even
	though this is the employee's own data, a chat transcript is exactly
	the kind of place a full PAN/UAN/bank account number shouldn't sit in
	plaintext (screenshots, synced chat history, shared devices) - the
	full number stays visible to HR in the actual Employee record."""
	value = str(value or "").strip()
	if not value:
		return None
	if len(value) <= keep:
		return value
	return "*" * (len(value) - keep) + value[-keep:]


def _pick_field(message, mapping):
	"""Bundle-style intents (profile, contact info, bank details, approvers,
	statutory IDs) default to showing everything in the category - but if
	the question names one specific thing ("my pan no?", "what is my full
	name"), that one thing should come back alone, not the whole bundle.
	mapping is an ordered list of (regex, field_key) pairs, most specific
	first; returns the first matching field_key, or None if the question
	was generic ("my profile", "my statutory ids") and the caller should
	show the full bundle instead."""
	msg = (message or "").lower()
	for pattern, key in mapping:
		if re.search(pattern, msg):
			return key
	return None


_PROFILE_FIELDS = [
	(r"\bname\b", "employee_name"),
	(r"designation", "designation"),
	(r"department", "department"),
	(r"employment type", "employment_type"),
	(r"\bbranch\b", "branch"),
	(r"\bgrade\b", "grade"),
	(r"employee (id|number|code)", "employee_number"),
	(r"date of joining|when.*join", "date_of_joining"),
	(r"date of appointment", "custom_date_of_appointment"),
	(r"retire", "date_of_retirement"),
	(r"\bstatus\b|am i active", "status"),
]
_PROFILE_LABELS = {
	"employee_name": _("Your name on file"),
	"designation": _("Designation"),
	"department": _("Department"),
	"employment_type": _("Employment Type"),
	"branch": _("Branch"),
	"grade": _("Grade"),
	"employee_number": _("Employee Number"),
	"date_of_joining": _("Date of Joining"),
	"custom_date_of_appointment": _("Date of Appointment"),
	"date_of_retirement": _("Date of Retirement"),
	"status": _("Employment Status"),
}
_PROFILE_DATE_FIELDS = ("date_of_joining", "custom_date_of_appointment", "date_of_retirement")


def _my_profile(employee, message=None):
	d = frappe.db.get_value("Employee", employee, list(_PROFILE_LABELS.keys()), as_dict=True)
	specific = _pick_field(message, _PROFILE_FIELDS)

	if specific:
		value = d.get(specific)
		if not value:
			return _("{0} isn't on file for you. Please check with HR.").format(_PROFILE_LABELS[specific])
		if specific in _PROFILE_DATE_FIELDS:
			value = formatdate(value)
		return _("{0}: {1}").format(_PROFILE_LABELS[specific], value)

	lines = []
	for key in ("designation", "department", "employment_type", "branch", "grade", "employee_number", "status"):
		if d.get(key):
			lines.append("{0}: {1}".format(_PROFILE_LABELS[key], d[key]))
	if d.get("date_of_joining"):
		lines.append(_("Date of Joining: {0}").format(formatdate(d["date_of_joining"])))
	if not lines:
		return _("Your profile doesn't have these details filled in yet. Please check with HR.")
	return _("Your profile:") + "<br>" + "<br>".join(lines)


_CONTACT_FIELDS = [
	(r"mobile|phone|cell", "cell_number"),
	(r"personal email", "personal_email"),
	(r"company email", "company_email"),
	(r"permanent address", "permanent_address"),
	(r"\baddress\b", "current_address"),
	(r"\bemail\b", "_email_generic"),
]
_CONTACT_LABELS = {
	"cell_number": _("Mobile"),
	"personal_email": _("Personal Email"),
	"company_email": _("Company Email"),
	"current_address": _("Current Address"),
	"permanent_address": _("Permanent Address"),
}


def _my_contact_info(employee, message=None):
	d = frappe.db.get_value("Employee", employee, list(_CONTACT_LABELS.keys()), as_dict=True)
	specific = _pick_field(message, _CONTACT_FIELDS)
	if specific == "_email_generic":
		# Bare "my email", no personal/company qualifier - prefer whichever is
		# actually on file, company email first.
		specific = "company_email" if d.get("company_email") else "personal_email"

	if specific:
		value = d.get(specific)
		if not value:
			return _("{0} isn't on file for you. Please check with HR.").format(_CONTACT_LABELS[specific])
		return _("{0}: {1}").format(_CONTACT_LABELS[specific], value)

	lines = [
		"{0}: {1}".format(_CONTACT_LABELS[k], d[k])
		for k in ("cell_number", "personal_email", "company_email", "current_address", "permanent_address")
		if d.get(k)
	]
	if not lines:
		return _("No contact details are on file for you yet. Please check with HR.")
	return _("Your contact details on file:") + "<br>" + "<br>".join(lines)


def _emergency_contact(employee, message=None):
	d = frappe.db.get_value(
		"Employee", employee, ["person_to_be_contacted", "emergency_phone_number", "relation"], as_dict=True
	)
	if not d.person_to_be_contacted and not d.emergency_phone_number:
		return _("No emergency contact is on file for you. Please share one with HR.")
	bits = [d.person_to_be_contacted or _("(name not on file)")]
	if d.relation:
		bits.append("({0})".format(d.relation))
	if d.emergency_phone_number:
		bits.append("- " + d.emergency_phone_number)
	return _("Your emergency contact on file: {0}").format(" ".join(bits))


def _probation_status(employee, message=None):
	d = frappe.db.get_value("Employee", employee, ["scheduled_confirmation_date", "final_confirmation_date"], as_dict=True)
	if d.final_confirmation_date:
		return _("You're already confirmed, as of {0}.").format(formatdate(d.final_confirmation_date))
	if d.scheduled_confirmation_date:
		return _("Your confirmation is scheduled for {0}.").format(formatdate(d.scheduled_confirmation_date))
	return _("No confirmation/probation date is set on your record. Please check with HR.")


def _contract_end(employee, message=None):
	end_date = frappe.db.get_value("Employee", employee, "contract_end_date")
	if not end_date:
		return _("You don't have a contract end date on record - if you're not on a fixed-term contract, that's expected.")
	return _("Your contract end date is {0}.").format(formatdate(end_date))


_APPROVER_FIELDS = [
	(r"leave.*approv", "leave_approver"),
	(r"expense.*approv", "expense_approver"),
	(r"shift.*approv", "shift_request_approver"),
]
_APPROVER_LABELS = {
	"leave_approver": _("Leave approver"),
	"expense_approver": _("Expense approver"),
	"shift_request_approver": _("Shift request approver"),
}


def _approvers(employee, message=None):
	d = frappe.db.get_value("Employee", employee, list(_APPROVER_LABELS.keys()), as_dict=True)
	specific = _pick_field(message, _APPROVER_FIELDS)

	if specific:
		user_id = d.get(specific)
		if not user_id:
			return _("No {0} is set on your employee record. Please check with HR.").format(_APPROVER_LABELS[specific].lower())
		name = frappe.db.get_value("User", user_id, "full_name") or user_id
		return _("Your {0}: {1}").format(_APPROVER_LABELS[specific].lower(), name)

	lines = []
	for field, label in _APPROVER_LABELS.items():
		user_id = d.get(field)
		if user_id:
			name = frappe.db.get_value("User", user_id, "full_name") or user_id
			lines.append("{0}: {1}".format(label, name))
	if not lines:
		return _("No approvers are set on your employee record. Please check with HR.")
	return _("Your approvers:") + "<br>" + "<br>".join(lines)


def _my_shift(employee, message=None):
	shift = frappe.db.get_value("Employee", employee, "default_shift")
	if not shift:
		return _("No default shift is set on your employee record.")
	return _("Your shift is {0}.").format(shift)


_BANK_FIELDS = [
	(r"ifsc", "ifsc_code"),
	(r"account (number|no)|a/?c no", "bank_ac_no"),
	(r"bank name|which bank", "bank_name"),
]
_BANK_LABELS = {"bank_name": _("Bank"), "bank_ac_no": _("Account"), "ifsc_code": _("IFSC")}


def _bank_details(employee, message=None):
	d = frappe.db.get_value("Employee", employee, list(_BANK_LABELS.keys()), as_dict=True)
	specific = _pick_field(message, _BANK_FIELDS)

	if specific:
		value = d.get(specific)
		if not value:
			return _("{0} isn't on file for you. Please check with HR.").format(_BANK_LABELS[specific])
		if specific == "bank_ac_no":
			value = _mask_tail(value)
		return _("{0}: {1}").format(_BANK_LABELS[specific], value)

	if not d.bank_name and not d.bank_ac_no:
		return _("No bank details are on file for you. Please check with HR.")
	lines = []
	if d.bank_name:
		lines.append(_("Bank: {0}").format(d.bank_name))
	if d.bank_ac_no:
		lines.append(_("Account: {0}").format(_mask_tail(d.bank_ac_no)))
	if d.ifsc_code:
		lines.append(_("IFSC: {0}").format(d.ifsc_code))
	return _("Your bank details on file (account number partly masked for safety):") + "<br>" + "<br>".join(lines)


_STATUTORY_FIELDS = [
	(r"\bpan\b", "pan_number"),
	(r"\buan\b", "custom_uan"),
	(r"provident fund|\bpf\b", "provident_fund_account"),
	(r"aadhaar", "custom_aadhaar_number"),
	(r"esic", "custom_esic_number"),
]
_STATUTORY_LABELS = {
	"pan_number": _("PAN"),
	"custom_uan": _("UAN"),
	"provident_fund_account": _("PF Account"),
	"custom_aadhaar_number": _("Aadhaar"),
	"custom_esic_number": _("ESIC Number"),
}


def _statutory_ids(employee, message=None):
	d = frappe.db.get_value("Employee", employee, list(_STATUTORY_LABELS.keys()), as_dict=True)
	specific = _pick_field(message, _STATUTORY_FIELDS)

	if specific:
		value = d.get(specific)
		if not value:
			return _("No {0} is on file for you yet. Please check with HR.").format(_STATUTORY_LABELS[specific])
		return _("Your {0}: {1}").format(_STATUTORY_LABELS[specific], _mask_tail(value))

	lines = []
	for field, label in _STATUTORY_LABELS.items():
		if d.get(field):
			lines.append("{0}: {1}".format(label, _mask_tail(d[field])))
	if not lines:
		return _("No statutory ID numbers are on file for you yet. Please check with HR.")
	return _("Your statutory IDs on file (masked for safety - HR can see the full numbers):") + "<br>" + "<br>".join(lines)


_SALARY_BREAKUP_FIELDS = [
	(r"ctc.*month|monthly ctc", "custom_ctcmonth"),
	(r"\bctc\b", "ctc"),
	(r"gross.*month|monthly gross", "custom_gross_salarymonth"),
	(r"gross", "custom_gross_salaryannum"),
	(r"net.*month|monthly.*(take.?home|net)|take.?home", "custom_net_salarymonth"),
	(r"net salary|net.*annum", "custom_net_salaryannum"),
	(r"deduct.*month", "custom_total_deductmonth"),
	(r"deduct", "custom_total_deductannum"),
	(r"allow.*month", "custom_total_allowmonth"),
	(r"allow", "custom_total_allowannum"),
	(r"benefit.*month", "custom_total_benefitmonth"),
	(r"benefit", "custom_total_benefitannum"),
]
_SALARY_BREAKUP_LABELS = {
	"ctc": _("CTC"),
	"custom_ctcmonth": _("CTC per month"),
	"custom_ctcannum": _("CTC per annum"),
	"custom_gross_salarymonth": _("Gross Salary per month"),
	"custom_gross_salaryannum": _("Gross Salary per annum"),
	"custom_net_salarymonth": _("Net Salary per month"),
	"custom_net_salaryannum": _("Net Salary per annum"),
	"custom_total_deductmonth": _("Total Deductions per month"),
	"custom_total_deductannum": _("Total Deductions per annum"),
	"custom_total_allowmonth": _("Total Allowances per month"),
	"custom_total_allowannum": _("Total Allowances per annum"),
	"custom_total_benefitmonth": _("Total Benefits per month"),
	"custom_total_benefitannum": _("Total Benefits per annum"),
}


def _salary_breakup(employee, message=None):
	# Same sensitivity tier as the payslip intent - this is the employee's
	# own compensation summary, not anyone else's.
	d = frappe.db.get_value("Employee", employee, list(_SALARY_BREAKUP_LABELS.keys()), as_dict=True)
	specific = _pick_field(message, _SALARY_BREAKUP_FIELDS)

	if specific:
		value = d.get(specific)
		if not value:
			return _("{0} isn't on file for you. Please check with HR.").format(_SALARY_BREAKUP_LABELS[specific])
		return _("Your {0}: {1}").format(_SALARY_BREAKUP_LABELS[specific], fmt_money(value))

	lines = [
		"{0}: {1}".format(_SALARY_BREAKUP_LABELS[k], fmt_money(d[k]))
		for k in ("ctc", "custom_gross_salaryannum", "custom_net_salaryannum", "custom_total_deductannum", "custom_total_allowannum")
		if d.get(k)
	]
	if not lines:
		return _("No salary breakup is on file for you yet. Please check with HR.")
	return _("Your salary breakup:") + "<br>" + "<br>".join(lines)


def _weekly_off(employee, message=None):
	d = frappe.db.get_value("Employee", employee, ["weekly_off_type", "fixed_off_day"], as_dict=True)
	if d.fixed_off_day:
		return _("Your weekly off is {0}.").format(d.fixed_off_day)
	if d.weekly_off_type:
		return _("Your weekly off type is: {0}.").format(d.weekly_off_type)
	return _("No weekly off is set on your employee record. Please check with HR.")


def _education(employee, message=None):
	rows = frappe.get_all(
		"Employee Education",
		filters={"parent": employee},
		fields=["qualification", "school_univ", "year_of_passing", "class_per"],
		order_by="idx asc",
	)
	if not rows:
		return _("No education details are on file for you yet. Please check with HR.")
	lines = [
		_("{0} - {1}{2}").format(
			r.qualification or _("(qualification not on file)"),
			r.school_univ or _("(institute not on file)"),
			", {0}".format(r.year_of_passing) if r.year_of_passing else "",
		)
		for r in rows
	]
	return _("Your education on file:") + "<br>" + "<br>".join(lines)


def _work_history(employee, message=None):
	rows = frappe.get_all(
		"Employee External Work History",
		filters={"parent": employee},
		fields=["company_name", "designation", "total_experience"],
		order_by="idx asc",
	)
	if not rows:
		return _("No previous work history is on file for you.")
	lines = [
		_("{0} - {1}").format(r.company_name or _("(company not on file)"), r.designation or "")
		for r in rows
	]
	return _("Your work history on file:") + "<br>" + "<br>".join(lines)


_PASSPORT_FIELDS = [
	(r"number", "passport_number"),
	(r"valid|expir", "valid_upto"),
	(r"issue date|date of issue", "date_of_issue"),
	(r"place of issue|where.*issued", "place_of_issue"),
]
_PASSPORT_LABELS = {
	"passport_number": _("Passport Number"),
	"valid_upto": _("Valid Until"),
	"date_of_issue": _("Date of Issue"),
	"place_of_issue": _("Place of Issue"),
}


def _passport_details(employee, message=None):
	d = frappe.db.get_value("Employee", employee, list(_PASSPORT_LABELS.keys()), as_dict=True)
	specific = _pick_field(message, _PASSPORT_FIELDS)

	if specific:
		value = d.get(specific)
		if not value:
			return _("{0} isn't on file for you. Please check with HR.").format(_PASSPORT_LABELS[specific])
		if specific in ("valid_upto", "date_of_issue"):
			value = formatdate(value)
		return _("Your {0}: {1}").format(_PASSPORT_LABELS[specific], value)

	if not d.passport_number:
		return _("No passport details are on file for you.")
	lines = [_("Passport Number: {0}").format(_mask_tail(d.passport_number))]
	if d.valid_upto:
		lines.append(_("Valid Until: {0}").format(formatdate(d.valid_upto)))
	if d.place_of_issue:
		lines.append(_("Place of Issue: {0}").format(d.place_of_issue))
	return _("Your passport details on file:") + "<br>" + "<br>".join(lines)


_PERSONAL_FIELDS = [
	(r"marital|married", "marital_status"),
	(r"blood group", "blood_group"),
	(r"health insurance", "health_insurance_no"),
	(r"health", "health_details"),
]
_PERSONAL_LABELS = {
	"marital_status": _("Marital Status"),
	"blood_group": _("Blood Group"),
	"health_details": _("Health Details"),
	"health_insurance_no": _("Health Insurance No"),
}


def _personal_details(employee, message=None):
	d = frappe.db.get_value("Employee", employee, list(_PERSONAL_LABELS.keys()), as_dict=True)
	specific = _pick_field(message, _PERSONAL_FIELDS)

	if specific:
		value = d.get(specific)
		if not value:
			return _("{0} isn't on file for you. Please check with HR.").format(_PERSONAL_LABELS[specific])
		return _("Your {0}: {1}").format(_PERSONAL_LABELS[specific], value)

	lines = [
		"{0}: {1}".format(_PERSONAL_LABELS[k], d[k])
		for k in ("marital_status", "blood_group", "health_details", "health_insurance_no")
		if d.get(k)
	]
	if not lines:
		return _("No personal details are on file for you yet.")
	return _("Your personal details on file:") + "<br>" + "<br>".join(lines)


# ---- HRMS how-to / config info (no Employee record needed) ----


def _leave_types_policy():
	rows = frappe.get_all(
		"Leave Type",
		fields=["name", "max_leaves_allowed", "is_carry_forward", "is_lwp"],
		order_by="name asc",
	)
	if not rows:
		return _("No leave types are configured in the system.")

	lines = []
	for r in rows:
		bits = []
		if r.max_leaves_allowed:
			bits.append(_("up to {0}/year").format(r.max_leaves_allowed))
		if r.is_carry_forward:
			bits.append(_("carries forward"))
		if r.is_lwp:
			bits.append(_("unpaid"))
		lines.append(r.name + (" (" + ", ".join(bits) + ")" if bits else ""))
	return _("Leave types configured in the system:") + "<br>" + "<br>".join(lines)


def _apply_leave():
	return _(
		"To apply for leave: use the search bar at the top and type "
		"\"New Leave Application\". Fill in the leave type, dates and reason, "
		"then submit - it routes to your reporting manager for approval."
	)


def _attendance_regularize():
	return _(
		"To fix a missed or wrong attendance entry: use the search bar at "
		"the top and type \"New Attendance Request\". Fill in the date range "
		"and reason, then submit - it goes to your reporting manager for "
		"approval."
	)


def _comp_off(employee, message=None):
	# Same gap as leave_balance/leave_taken - "on which date I taken
	# compoff" shares the same intent as the how-to question, but wants
	# real data (when it was actually taken), not instructions on how to
	# request it. Caught by real testing.
	if re.search(r"\btaken\b|\bused\b|\bavailed\b|\bwhich date\b|\bwhat date\b", (message or "").lower()):
		rows = frappe.get_all(
			"Leave Application",
			filters={"employee": employee, "leave_type": "Comp Off", "status": "Approved", "docstatus": 1},
			fields=["from_date", "to_date"],
			order_by="from_date desc",
			limit=5,
		)
		if not rows:
			return _("You haven't taken any Comp Off yet.")
		lines = [
			formatdate(r.from_date) if r.from_date == r.to_date
			else _("{0} to {1}").format(formatdate(r.from_date), formatdate(r.to_date))
			for r in rows
		]
		return _("You've taken Comp Off on:") + "<br>" + "<br>".join(lines)

	return _(
		"To claim compensatory off for working a holiday/weekend: use the "
		"search bar at the top and type \"New Compensatory Leave Request\". "
		"Once approved, it's added to your Comp Off balance."
	)


def _expense_claim(employee, message=None):
	# Same how-to/data split as comp_off - "status of my expense claim"
	# needs the real record, not instructions on how to file one.
	if re.search(r"\bstatus\b|\btaken\b|\bhow much\b|\bpending\b", (message or "").lower()):
		rows = frappe.get_all(
			"Expense Claim",
			filters={"employee": employee, "docstatus": ["!=", 2]},
			fields=["name", "posting_date", "total_claimed_amount", "status"],
			order_by="posting_date desc",
			limit=5,
		)
		if not rows:
			return _("You haven't filed any expense claims yet.")
		lines = [
			_("{0}: {1} - {2}").format(formatdate(r.posting_date), fmt_money(r.total_claimed_amount), r.status)
			for r in rows
		]
		return _("Your recent expense claims:") + "<br>" + "<br>".join(lines)

	return _(
		"To raise a reimbursement: use the search bar at the top and type "
		"\"New Expense Claim\". Add the expense items and attach receipts, "
		"then submit for approval."
	)


def _onboarding_docs():
	return _(
		"Onboarding documents are uploaded by the candidate through their "
		"own secure portal link (sent by HR from the Job Offer) before they "
		"join. As an existing employee you won't need to re-upload these - "
		"contact HR if something's missing from your record."
	)


def _greeting():
	return _(
		"Hi! I can look up your leave balance, attendance, payslip, holidays "
		"and more, or walk you through how to apply for leave, regularize "
		"attendance, claim comp off or raise an expense claim. Type "
		"\"help\" anytime to see the full list."
	)


def _thanks():
	return _("You're welcome! Let me know if there's anything else I can help with.")


def _help(*args):
	return _(
		"I can answer things like:<br>"
		"- \"What's my leave balance?\"<br>"
		"- \"Status of my leave request\"<br>"
		"- \"My attendance this month\"<br>"
		"- \"My latest payslip\"<br>"
		"- \"When's the next holiday?\"<br>"
		"- \"Who is my manager?\"<br>"
		"- \"What's my notice period?\"<br>"
		"- \"What leave types are there?\""
		"<br><br>"
		"Or how-to questions like:<br>"
		"- \"How do I apply for leave?\"<br>"
		"- \"How do I regularize attendance?\"<br>"
		"- \"How do I claim comp off?\"<br>"
		"- \"How do I raise an expense claim?\"<br>"
		"- \"I want to resign, what do I do?\""
		"<br><br>"
		"Or about your own profile - designation, department, contact "
		"details, emergency contact, bank details, PAN/UAN/ESIC, CTC and "
		"salary breakup, probation/confirmation date, contract end date, "
		"your shift/weekly off, education, work history, passport, "
		"personal/health details, or your leave/expense/shift approvers."
	)


# Checked in order - more specific patterns before broader/catch-all ones,
# e.g. "how do I apply for leave" must hit _apply_leave (how-to) before the
# generic leave-status intent, and "regularize attendance" must hit the
# how-to intent before the plain attendance data lookup. Same order is used
# to break ties in the fuzzy fallback (see _fuzzy_match). "keywords" is the
# vocabulary that fallback fuzzy-matches typo'd words against - greeting/
# thanks/help are deliberately excluded (kept regex-only, short phrases
# where fuzzy matching would misfire more than it'd help). "key" is the
# stable name the optional AI fallback classifies into (see _ai_classify) -
# also excluded for greeting/thanks/help, no point spending an API call to
# learn someone said "hi".
def _entry(key, pattern, handler, needs_employee, keywords=None):
	return {
		"key": key,
		"pattern": re.compile(pattern, re.I),
		"handler": handler,
		"needs_employee": needs_employee,
		"keywords": keywords or [],
	}


INTENTS = [
	_entry(None, r"^\s*(hi+|hello+|hey+|good (morning|afternoon|evening))\b", _greeting, False),
	_entry(None, r"^\s*(thanks|thank you|thx|ty)\b", _thanks, False),
	_entry("leave_balance", r"leave.*balance|balance.*leave|how many leave", _leave_balance, True, ["leave", "balance", "many", "left", "remaining"]),
	_entry("apply_leave", r"apply.*leave|leave.*apply|(how|new|fill|raise|create).*leave.*application", _apply_leave, False, ["apply", "leave", "application", "new"]),
	_entry("leave_types_policy", r"leave (type|polic)|types? of leave|what leaves", _leave_types_policy, False, ["leave", "type", "types", "policy", "policies"]),
	_entry("leave_status", r"leave.*(status|request|application)", _leave_status, True, ["leave", "status", "request", "application"]),
	# Deliberately excludes "attendance" from its own keyword list - it's
	# shared with the plain _attendance lookup below, and would win fuzzy
	# ties for any typo'd "attendance" even without "regularize"/"correct"
	# present (this misfired on "atendance this month" during testing).
	_entry("attendance_regularize", r"regulari.*attendance|attendance.*regulari|correct.*attendance|attendance.*request", _attendance_regularize, False, ["regularize", "regularise", "correct"]),
	_entry("attendance", r"attendance|present|absent", _attendance, True, ["attendance", "present", "absent"]),
	# Checked before payslip - "my salary account" (a real, common Indian
	# usage meaning "which bank account my salary goes to") would otherwise
	# be caught by payslip's bare "my salary" pattern below.
	_entry("bank_details", r"bank (details|account|name)|salary account|account number|which bank|\bifsc\b", _bank_details, True, ["bank", "ifsc"]),
	# Checked before payslip too - "what's my ctc"/"gross salary"/"net
	# salary" are about the compensation summary on the Employee record
	# itself, not a specific month's Salary Slip.
	_entry("salary_breakup", r"\bctc\b|gross salary|net salary|take.?home|total (deduct|allow|benefit)", _salary_breakup, True, ["ctc"]),
	_entry("payslip", r"pay ?slip|salary slip|my salary", _payslip, True, ["payslip", "salary", "slip"]),
	_entry("comp_off", r"comp ?off|compensatory", _comp_off, True, ["comp", "compensatory"]),
	_entry("expense_claim", r"expense claim|reimbursement", _expense_claim, True, ["expense", "claim", "reimbursement"]),
	_entry("onboarding_docs", r"onboarding|document.*upload|upload.*document", _onboarding_docs, False, ["onboarding", "document", "documents", "upload"]),
	_entry("notice_period", r"notice period|notice days", _notice_period, True, ["notice", "period", "days"]),
	_entry(
		"resignation",
		r"resign|resignation|quit\b|exit (the )?company|leave (the )?(company|job|organi[sz]ation)",
		_resignation,
		True,
		["resign", "resignation", "quit"],
	),
	# Checked before my_contact_info - "who is my emergency contact" would
	# otherwise be ambiguous with a generic contact-details query.
	_entry("emergency_contact", r"emergency contact|emergency phone", _emergency_contact, True, ["emergency"]),
	_entry("my_contact_info", r"my (mobile|phone|personal email|company email|email|address)\b|contact (details|info)", _my_contact_info, True, ["mobile", "phone", "address"]),
	_entry("probation_status", r"probation|confirmation date|when.*(i be )?confirm", _probation_status, True, ["probation", "confirmation"]),
	_entry("contract_end", r"contract end|when.*(does )?my contract", _contract_end, True, ["contract"]),
	_entry("approvers", r"who approves|my (leave|expense|shift) approver", _approvers, True, ["approver", "approves"]),
	_entry("my_shift", r"my shift|which shift|what shift", _my_shift, True, ["shift"]),
	_entry("weekly_off", r"weekly off|fixed off day|which day.*off", _weekly_off, True, ["weekly"]),
	_entry("statutory_ids", r"\bpan\b|\buan\b|aadhaar|provident fund account|\bpf number\b|\besic\b", _statutory_ids, True, ["pan", "uan", "aadhaar", "esic"]),
	_entry(
		"my_profile",
		r"my (profile|designation|department|branch|(full )?name|employee (id|number|code))|which department|what.*my (designation|name)|date of appointment|when.*i retire|am i active",
		_my_profile,
		True,
		["designation", "department", "branch", "profile"],
	),
	_entry("education", r"my education|my qualification|which university|where.*i stud", _education, True, ["education", "qualification"]),
	_entry("work_history", r"work history|previous compan|earlier job|past employ", _work_history, True, ["history"]),
	_entry("passport_details", r"passport", _passport_details, True, ["passport"]),
	_entry("personal_details", r"marital status|blood group|health insurance|health details|am i married", _personal_details, True, ["marital", "blood"]),
	_entry("holiday", r"holiday", _next_holiday, True, ["holiday", "holidays"]),
	_entry("manager", r"manager|report(s|ing)? ?to", _manager, True, ["manager", "reports", "reporting"]),
	_entry(None, r"help|what can you|commands", _help, False),
]

INTENTS_BY_KEY = {entry["key"]: entry for entry in INTENTS if entry["key"]}
