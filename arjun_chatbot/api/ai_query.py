# arjun_chatbot/api/ai_query.py
#
# The AI-first answer path for the HR chatbot: instead of a fixed table of
# regex intents (see hr_chatbot.py, which this module leaves completely
# untouched and which ask() still falls back to whenever this module
# returns None), the LLM is given three generic, read-only tools and picks
# which one(s) to call and with what doctype/fields/filters. The security
# boundary is NOT the LLM's judgement - it is _scope_filters() below, which
# runs on every tool call and unconditionally overwrites whatever
# employee/user/parent filter the model supplied with the real one resolved
# server-side from frappe.session.user, the same way hr_chatbot._current_
# employee() already does. The model never sees another employee's data
# even if it tries to ask for it - the forced filter combines with any
# other filters via AND, so it can only ever narrow the result, never
# widen it past the caller's own records.
#
# Doctype access is further limited to a fixed HR/payroll allow-list
# (ALLOWED_DOCTYPES below) - broad within HR, but deliberately not "every
# doctype in the ERPNext install" (Accounting/Stock/Selling etc. stay out
# of reach). Every read still goes through frappe.get_list as the real
# session user (never ignore_permissions=True), so Frappe's own role/user-
# permission layer is a second, independent check underneath the forced
# filter, not a replacement for it.

import json

import frappe
from frappe.utils import flt

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_TIMEOUT = 15
_MAX_TOOL_ROUNDS = 4
_QUERY_LIMIT_MAX = 50
_AGGREGATE_ROW_CAP = 1000

# Fields shown only partly masked (see _mask_row) - same tier of sensitive
# identifiers hr_chatbot.py's own fixed handlers already mask via
# _mask_tail (bank account, PAN, UAN, Aadhaar, ESIC, PF, passport).
_MASKED_FIELDS = {
	"bank_ac_no",
	"pan_number",
	"custom_uan",
	"custom_aadhaar_number",
	"custom_esic_number",
	"provident_fund_account",
	"passport_number",
}

# doctype -> (scope mode, one-line description for the system prompt).
# Scope modes:
#   "self"             - the doctype IS Employee; scoped by name=<employee>
#   "employee_field"   - has an "employee" link field; scoped by that
#   "child_of_employee" - child table under Employee; scoped by parent=<employee>
#   "global"           - shared reference data, not personally owned; unscoped
ALLOWED_DOCTYPES = {
	"Employee": ("self", "profile - employee_name, designation, dept, contact, bank, statutory IDs (PAN/UAN/PF/Aadhaar/ESIC - 'no'/'num'/'number' all mean the same thing), CTC, personal/family fields (father's name etc.) - call list_fields if unsure"),
	"Leave Application": ("employee_field", "leave requests - leave_type, from_date, to_date, status, total_leave_days, half_day"),
	"Leave Allocation": ("employee_field", "leave allocated per type/period"),
	"Leave Type": ("global", "leave types - max_leaves_allowed, is_carry_forward, is_lwp"),
	"Attendance": ("employee_field", "daily attendance - attendance_date, status"),
	"Attendance Request": ("employee_field", "attendance regularization requests"),
	"Salary Slip": ("employee_field", "payslips - start_date, end_date, net_pay, gross_pay, status"),
	"Salary Detail": ("employee_field", "payslip line items (incl. bonus) - salary_component, amount, parent=Salary Slip"),
	"Expense Claim": ("employee_field", "reimbursement claims - posting_date, total_claimed_amount, status"),
	"Compensatory Leave Request": ("employee_field", "comp-off requests - work_from_date, work_end_date"),
	"Employee Loan": ("employee_field", "loans - loan_type, principal_amount, emi_amount, status, outstanding_balance"),
	"Employee Education": ("child_of_employee", "education history (child table)"),
	"Employee External Work History": ("child_of_employee", "previous-employer history (child table)"),
	"Holiday List": ("global", "holiday calendars"),
	"Holiday": ("global", "holidays - holiday_date, description, parent=Holiday List"),
	"Department": ("global", "company departments"),
	"Designation": ("global", "job designations"),
	"Shift Type": ("global", "shift definitions - start_time, end_time"),
	"Job Opening": ("global", "public open positions, not employee-scoped - job_title, status, vacancies, posted_on, closes_on"),
}

# doctype -> a sensible default order_by when the model doesn't specify one
# or specifies an invalid one. Frappe's own default (modified desc - last
# edited, not chronological) produced real, confirmed-wrong answers for
# date-scoped questions like "my attendance this month" - with no
# doctype-appropriate ordering, a plain LIMIT can silently return rows
# scattered across unrelated months instead of the recent/relevant ones.
_DEFAULT_ORDER_BY = {
	"Attendance": "attendance_date desc",
	"Attendance Request": "from_date desc",
	"Leave Application": "from_date desc",
	"Leave Allocation": "from_date desc",
	"Salary Slip": "end_date desc",
	"Expense Claim": "posting_date desc",
	"Compensatory Leave Request": "work_from_date desc",
	"Employee Loan": "disbursement_date desc",
	"Holiday": "holiday_date asc",
}


def answer(message):
	"""Entry point called from hr_chatbot.ask(). Returns a reply string, or
	None if AI fallback is off/unconfigured, the call failed, or the model
	never produced a final answer within the round cap - callers must treat
	None as "fall back to the existing fixed-intent pipeline", never as an
	error to surface."""
	settings = frappe.get_cached_doc("HR Chatbot Settings")
	if not settings.get("enable_ai_fallback"):
		return None

	from frappe.utils.password import get_decrypted_password

	api_key = get_decrypted_password("HR Chatbot Settings", "HR Chatbot Settings", "groq_api_key", raise_exception=False)
	if not api_key:
		return None

	try:
		return _run_tool_loop(message, api_key, settings.get("groq_model") or "openai/gpt-oss-20b")
	except Exception:
		frappe.log_error(title="HR chatbot AI query failed")
		return None


def _run_tool_loop(message, api_key, model):
	messages = [
		{"role": "system", "content": _system_prompt()},
		{"role": "user", "content": message},
	]

	for round_index in range(_MAX_TOOL_ROUNDS):
		# tool_choice="required" was tried here to force grounding on round
		# 0 (a real hallucination - a fabricated "+1234567890" phone number
		# that doesn't exist anywhere in the DB - showed the model will
		# skip tools when left fully to "auto"). Reverted: Groq treats
		# "required" as a hard constraint and 400s the WHOLE request if the
		# model doesn't comply, and the fallback-retry-on-400 this caused
		# added a full extra network round trip on every single message -
		# confirmed via a real request that took 120+ seconds end to end
		# for a WhatsApp user waiting on a reply. A slow/silent bot is worse
		# than the occasional hallucination the strengthened system prompt
		# (see the "RULE, never break it" grounding instruction above)
		# already substantially reduces on its own.
		response = _post_with_retry(api_key, model, messages, "auto")
		response.raise_for_status()
		choice = response.json()["choices"][0]["message"]

		tool_calls = choice.get("tool_calls")
		if not tool_calls:
			content = (choice.get("content") or "").strip()
			return content or None

		# Groq's API requires the assistant message that requested the
		# tool calls to be echoed back before the tool results.
		messages.append(choice)
		for call in tool_calls:
			name = call["function"]["name"]
			try:
				args = json.loads(call["function"].get("arguments") or "{}")
			except Exception:
				args = {}
			result = _execute_tool(name, args)
			messages.append(
				{
					"role": "tool",
					"tool_call_id": call["id"],
					"name": name,
					"content": json.dumps(result, default=str),
				}
			)

	return None


def _post_with_retry(api_key, model, messages, tool_choice="auto"):
	"""One retry with a short backoff, but ONLY on a transient server error
	(5xx) - NOT on 429. This Groq key's actual limit (confirmed via its own
	response headers) is 8,000 TOKENS/minute, not a request count; a 429
	means that minute's budget is already spent, and retrying into it after
	1-2s just burns another request without a realistic chance of success
	until the window resets (~seconds to under a minute) - better to fail
	fast here and let the caller fall back to the fixed-intent pipeline
	immediately than make a person wait on a retry that can't work yet."""
	import time

	import requests

	last_response = None
	for attempt in range(2):
		if attempt:
			time.sleep(1.5)
		response = requests.post(
			_GROQ_URL,
			headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
			json={
				"model": model,
				"messages": messages,
				"tools": _TOOL_SCHEMAS,
				"tool_choice": tool_choice,
				"temperature": 0,
				"reasoning_effort": "low",
				"max_tokens": 400,
			},
			timeout=_GROQ_TIMEOUT,
		)
		last_response = response
		if response.status_code not in (500, 502, 503, 504):
			return response  # success, or a non-retryable failure (incl. 429)
	return last_response


def _system_prompt():
	from frappe.utils import get_first_day, get_last_day, nowdate

	today = nowdate()
	m_start, m_end = get_first_day(today), get_last_day(today)
	doctype_lines = "\n".join(f"- {dt}: {desc}" for dt, (_scope, desc) in ALLOWED_DOCTYPES.items())
	# Kept deliberately terse - this whole prompt is resent on every tool-
	# calling round (see _run_tool_loop), and this Groq key's free tier is
	# capped at 8,000 TOKENS/minute (confirmed via response headers), not
	# request count - a verbose prompt times a multi-round question was the
	# real cause of real 429s seen in testing, not call volume.
	return (
		"RULE, never break it: every value you state (number, date, ID, status, name) must come from "
		"a tool result in THIS conversation. Never answer a data question from memory/general "
		"knowledge and never invent a plausible-looking example value - if you haven't queried the "
		"field yet, call query_records for it first, every time, even for a question that looks "
		"simple. If the field/record genuinely doesn't exist or is empty, say so plainly - that is "
		"different from not having checked yet.\n"
		f"HR assistant for this HRMS. Today: {today}. 'This month' = {m_start} to {m_end} - always "
		"pass explicit date-range filters for date-scoped questions, never rely on limit/order_by "
		"alone to imply a period.\n"
		"'name' on any doctype is the internal document ID (e.g. Employee ID 'ADMIN-001'), NEVER a "
		"person's real name - use employee_name/first_name/last_name for that.\n"
		"Every tool is auto-scoped server-side to the logged-in employee, ignoring any employee/name "
		"the question mentions - you never need to ask who they are. ONLY if the question explicitly "
		"names a DIFFERENT specific person (e.g. 'salary of Mayank', 'John's leave balance') say you "
		"can only show the asker's own data instead of answering for that other name. A plain "
		"first-person question ('my phone number', 'my PAN') names no one else - just query and "
		"answer it directly, no disclaimer, no need to mention other employees at all.\n"
		"Doctypes:\n" + doctype_lines + "\n"
		"Call list_fields if unsure a doctype has a field (esp. personal-detail questions) before "
		"calling something off-topic. query_records for lookups, aggregate_records for sum/avg/count/"
		"min/max. No records found -> say so plainly, don't guess and don't deflect to 'check the "
		"portal yourself' when you're able to just look it up and answer.\n"
		"Only include a link in TWO cases: (1) a payslip - call get_document_pdf_url and use its "
		"exact 'url' in an <a> tag; (2) a self-service how-to (apply leave/regularize attendance/"
		"comp-off/expense claim) - call get_new_record_url and link it the same way. For every other "
		"question (status/data lookups like attendance, leave requests, profile fields, etc.) just "
		"answer in plain text with no link at all - most answers need none. NEVER write a URL "
		"yourself (e.g. never write example.com or invent a /desk#... "
		"path) - the only URLs you may ever output are the exact 'url' values a tool call just "
		"returned, copied verbatim.\n"
		"Short replies, <br> not markdown lists. Not HR-related at all -> say you can only help with "
		"HR questions."
	)


_TOOL_SCHEMAS = [
	{
		"type": "function",
		"function": {
			"name": "list_fields",
			"description": "List the queryable fields on an allowed HR doctype.",
			"parameters": {
				"type": "object",
				"properties": {"doctype": {"type": "string", "enum": list(ALLOWED_DOCTYPES)}},
				"required": ["doctype"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "query_records",
			"description": (
				"Fetch records from an allowed HR doctype, always scoped to the logged-in "
				"employee automatically. filters is a dict of fieldname -> value for equality, "
				"or fieldname -> [operator, value] for comparisons, e.g. "
				'{"attendance_date": [">=", "2026-01-01"]}.'
			),
			"parameters": {
				"type": "object",
				"properties": {
					"doctype": {"type": "string", "enum": list(ALLOWED_DOCTYPES)},
					"fields": {"type": "array", "items": {"type": "string"}},
					"filters": {"type": "object"},
					"order_by": {"type": "string", "description": "e.g. 'modified desc'"},
					"limit": {"type": "integer"},
				},
				"required": ["doctype"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "aggregate_records",
			"description": (
				"Compute sum/avg/count/min/max of one numeric field across matching records "
				"of an allowed HR doctype, scoped to the logged-in employee automatically."
			),
			"parameters": {
				"type": "object",
				"properties": {
					"doctype": {"type": "string", "enum": list(ALLOWED_DOCTYPES)},
					"field": {"type": "string"},
					"aggregate": {"type": "string", "enum": ["sum", "avg", "count", "min", "max"]},
					"filters": {"type": "object"},
				},
				"required": ["doctype", "field", "aggregate"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "get_document_pdf_url",
			"description": (
				"Build a downloadable-PDF link for a payslip (Salary Slip), using the company's own "
				"branded print format. Only ever used for Salary Slip - not for any other doctype."
			),
			"parameters": {
				"type": "object",
				"properties": {
					"doctype": {"type": "string", "enum": list(ALLOWED_DOCTYPES)},
					"name": {"type": "string", "description": "The record's 'name' (ID) field value."},
				},
				"required": ["doctype", "name"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "get_new_record_url",
			"description": (
				"Build a clickable link to the 'create new' form for a self-service doctype "
				"(Leave Application, Attendance Request, Compensatory Leave Request, Expense "
				"Claim), to include when explaining how to apply/request/claim something."
			),
			"parameters": {
				"type": "object",
				"properties": {"doctype": {"type": "string", "enum": list(ALLOWED_DOCTYPES)}},
				"required": ["doctype"],
			},
		},
	},
]


def _execute_tool(name, args):
	"""Dispatch one tool call. Never raises - any failure (bad doctype,
	bad field, DB error) comes back as {"error": ...} so the model can
	adjust and the loop keeps going, and internal exception text is never
	echoed back verbatim (logged instead)."""
	try:
		if name == "list_fields":
			return _tool_list_fields(args.get("doctype"))
		if name == "query_records":
			return _tool_query_records(
				args.get("doctype"), args.get("fields"), args.get("filters"), args.get("order_by"), args.get("limit")
			)
		if name == "aggregate_records":
			return _tool_aggregate_records(args.get("doctype"), args.get("field"), args.get("aggregate"), args.get("filters"))
		if name == "get_document_pdf_url":
			return _tool_get_document_pdf_url(args.get("doctype"), args.get("name"))
		if name == "get_new_record_url":
			return _tool_get_new_record_url(args.get("doctype"))
		return {"error": f"Unknown tool '{name}'."}
	except Exception:
		frappe.log_error(title="HR chatbot AI tool execution failed")
		return {"error": "Something went wrong running that lookup."}


def _current_employee():
	from arjun_chatbot.api.hr_chatbot import _current_employee as impl

	return impl()


def _mask_tail(value):
	from arjun_chatbot.api.hr_chatbot import _mask_tail as impl

	return impl(value)


def _slug(doctype):
	"""Frappe's own desk route slug - lowercase, spaces to hyphens. Matches
	the hand-written link paths hr_chatbot.py's fixed handlers already use
	(e.g. '/app/leave-application/new', '/app/compensatory-leave-request/new')."""
	return doctype.lower().replace(" ", "-")


def _tool_get_document_pdf_url(doctype, name):
	"""A downloadable-PDF link, using Frappe's own default print format for
	the doctype (e.g. this HRMS's branded payslip layout for Salary Slip -
	confirmed configured as Salary Slip's actual default_print_format, not
	hardcoded here so it keeps working if that format is ever renamed).
	Goes through frappe.utils.print_format.download_pdf, the same
	whitelisted endpoint the desk's own Print/PDF button uses - it still
	enforces the caller's real print permission on that specific document,
	so this grants no more access than any other scoped query already implied."""
	err = _check_doctype(doctype)
	if err:
		return err
	if not name:
		return {"error": "A record name/ID is required."}
	from urllib.parse import quote

	return {"url": f"/api/method/frappe.utils.print_format.download_pdf?doctype={quote(doctype, safe='')}&name={quote(str(name), safe='')}"}


def _tool_get_new_record_url(doctype):
	err = _check_doctype(doctype)
	if err:
		return err
	return {"url": f"/app/{_slug(doctype)}/new"}


def _check_doctype(doctype):
	if doctype not in ALLOWED_DOCTYPES:
		return {"error": f"'{doctype}' isn't an allowed doctype. Allowed: {', '.join(ALLOWED_DOCTYPES)}."}
	return None


_SKIP_FIELDTYPES = ("Section Break", "Column Break", "Tab Break", "HTML", "Button", "Table")


def _ordered_fieldnames(doctype):
	"""Fieldnames in the doctype's own natural field_order (how the form
	itself is laid out - e.g. Employee starts naming_series, first_name,
	middle_name, last_name, employee_name...), NOT alphabetical. Used for
	the "no fields specified" default in _tool_query_records - alphabetical
	order was a real, confirmed bug: it buries meaningful fields like
	employee_name behind a wall of custom_ fields that happen to sort
	earlier, so a vague query would silently return a well-formed but
	useless response instead of the fields a person would actually mean."""
	meta = frappe.get_meta(doctype)
	names = [f.fieldname for f in meta.fields if f.fieldtype not in _SKIP_FIELDTYPES]
	for extra in ("name", "owner", "creation", "modified", "idx"):
		if extra not in names:
			names.append(extra)
	return names


def _allowed_fields(doctype):
	return set(_ordered_fieldnames(doctype))


def _tool_list_fields(doctype):
	err = _check_doctype(doctype)
	if err:
		return err
	meta = frappe.get_meta(doctype)
	fields = [
		{"fieldname": f.fieldname, "label": f.label, "fieldtype": f.fieldtype}
		for f in meta.fields
		if f.fieldtype not in _SKIP_FIELDTYPES
	]
	# "name" is Frappe's internal document ID field, not part of meta.fields -
	# labelled loudly here since a model reading only this list (not the
	# system prompt) must not mistake it for a person's actual name.
	fields.append({"fieldname": "name", "label": "Document/Record ID (system-generated, NOT a person's name)", "fieldtype": "Data"})
	scope, _desc = ALLOWED_DOCTYPES[doctype]
	return {"doctype": doctype, "scope": scope, "fields": fields}


def _scope_filters(doctype, filters, employee):
	"""The security boundary: unconditionally overwrites whatever
	employee/parent-identifying filter the model supplied with the real
	one resolved server-side. Combined with any other filters via AND (how
	frappe.get_list filters always combine), so extra filters can only
	narrow the result further, never widen it past this employee's own
	records - see module docstring."""
	filters = dict(filters or {})
	scope, _desc = ALLOWED_DOCTYPES[doctype]
	if scope == "self":
		filters["name"] = employee
	elif scope == "employee_field":
		filters["employee"] = employee
	elif scope == "child_of_employee":
		filters["parent"] = employee
	# "global": shared reference data, left as the model specified it.
	return filters


def _validate_order_by(order_by, allowed_fields):
	import re

	if not order_by:
		return None
	match = re.match(r"^([A-Za-z0-9_]+)\s+(asc|desc)$", order_by.strip(), re.I)
	if not match or match.group(1) not in allowed_fields:
		return None
	return f"{match.group(1)} {match.group(2)}"


def _mask_row(row):
	for key in list(row.keys()):
		if key in _MASKED_FIELDS and row[key]:
			row[key] = _mask_tail(row[key])
	return row


def _tool_query_records(doctype, fields, filters, order_by, limit):
	err = _check_doctype(doctype)
	if err:
		return err

	employee = _current_employee()
	if not employee:
		return {"error": "No Employee record is linked to your account."}

	allowed_fields = _allowed_fields(doctype)
	fields = [f for f in (fields or []) if f in allowed_fields]
	if not fields:
		# A reasonable default so the model always gets something useful
		# back even if it didn't specify fields on the first try - natural
		# field order, not alphabetical (see _ordered_fieldnames).
		fields = _ordered_fieldnames(doctype)[:12]
	if "name" not in fields:
		fields = ["name"] + fields

	scoped_filters = _scope_filters(doctype, filters, employee)
	safe_order_by = _validate_order_by(order_by, allowed_fields) or _DEFAULT_ORDER_BY.get(doctype)
	safe_limit = max(1, min(int(limit or 30), _QUERY_LIMIT_MAX))

	rows = frappe.get_list(
		doctype,
		filters=scoped_filters,
		fields=fields,
		order_by=safe_order_by,
		limit_page_length=safe_limit,
		user=frappe.session.user,
	)
	rows = [_mask_row(r) for r in rows]
	return {"doctype": doctype, "count": len(rows), "records": rows}


def _tool_aggregate_records(doctype, field, aggregate, filters):
	err = _check_doctype(doctype)
	if err:
		return err
	if aggregate not in ("sum", "avg", "count", "min", "max"):
		return {"error": f"'{aggregate}' isn't a supported aggregate."}

	employee = _current_employee()
	if not employee:
		return {"error": "No Employee record is linked to your account."}

	allowed_fields = _allowed_fields(doctype)
	if field not in allowed_fields:
		return {"error": f"'{field}' isn't a field on {doctype}."}

	scoped_filters = _scope_filters(doctype, filters, employee)
	rows = frappe.get_list(
		doctype,
		filters=scoped_filters,
		fields=[field],
		limit_page_length=_AGGREGATE_ROW_CAP,
		user=frappe.session.user,
	)

	if aggregate == "count":
		return {"doctype": doctype, "field": field, "aggregate": aggregate, "result": len(rows), "record_count": len(rows)}

	values = [flt(r.get(field)) for r in rows if r.get(field) is not None]
	if not values:
		return {"doctype": doctype, "field": field, "aggregate": aggregate, "result": None, "record_count": len(rows)}

	result = {
		"sum": sum(values),
		"avg": sum(values) / len(values),
		"min": min(values),
		"max": max(values),
	}[aggregate]
	return {"doctype": doctype, "field": field, "aggregate": aggregate, "result": result, "record_count": len(rows)}
