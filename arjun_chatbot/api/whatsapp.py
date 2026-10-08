# arjun_chatbot/api/whatsapp.py
#
# WhatsApp channel for ALIA (see hr_chatbot.py), via Meta's WhatsApp Cloud
# API. This file is deliberately thin: it does identity resolution (which
# Employee is this WhatsApp number?) and HTML<->plain-text conversion, then
# hands the actual question to the exact same hr_chatbot.ask() the desk
# widget already calls. Every existing safeguard - employee-scoped AI tool
# calls, masking, the AI-first/fixed-intent fallback - applies unmodified,
# because the security boundary those rely on is frappe.session.user, and
# this file's whole job is to set that correctly (via frappe.set_user())
# before calling ask(), then hand off.
#
# No OTP step: a WhatsApp message is already authenticated by Meta as
# coming from that real phone number (unlike caller ID, it isn't
# spoofable), so identity is resolved by matching the sender's number
# against Employee.cell_number on file. Zero or multiple matches both
# refuse rather than guess - see _resolve_employee_by_phone.

import hashlib
import hmac
import html
import re

import frappe
from werkzeug.wrappers import Response

_GRAPH_API_VERSION = "v20.0"
_SEND_TIMEOUT = 15
# WhatsApp/Meta can redeliver the same webhook event - skip anything
# already handled recently rather than double-processing/double-replying.
_DEDUPE_TTL_SECONDS = 10 * 60


def _settings():
	return frappe.get_cached_doc("HR Chatbot Settings")


@frappe.whitelist(allow_guest=True)
def webhook(**kwargs):
	"""Single endpoint for both halves of Meta's webhook contract: GET is
	the one-time verification handshake, POST is every incoming message
	event afterward. kwargs (not named params) because Meta's GET query
	params are dotted ('hub.mode' etc.), not valid Python identifiers."""
	if frappe.request.method == "GET":
		return _handle_verification()
	return _handle_incoming()


def _handle_verification():
	"""Meta calls this once when you save the webhook URL in its dashboard,
	to prove you control this endpoint. Must echo back hub.challenge as the
	exact, raw response body (not JSON-wrapped) - returning a werkzeug
	Response directly bypasses frappe's usual {"message": ...} envelope,
	see frappe.handler.handle()'s isinstance(data, Response) passthrough."""
	settings = _settings()
	mode = frappe.form_dict.get("hub.mode")
	token = frappe.form_dict.get("hub.verify_token")
	challenge = frappe.form_dict.get("hub.challenge")

	expected_token = settings.get("whatsapp_verify_token")
	if mode == "subscribe" and expected_token and token == expected_token:
		return Response(str(challenge or ""), mimetype="text/plain", status=200)

	frappe.local.response.http_status_code = 403
	return Response("Verification token mismatch", mimetype="text/plain", status=403)


def _handle_incoming():
	settings = _settings()
	if not settings.get("enable_whatsapp"):
		# Quietly ack - do not reveal to an unauthenticated caller whether
		# the channel is even configured.
		return {"status": "ignored"}

	raw_body = frappe.request.get_data()
	if not _verify_signature(raw_body, settings):
		frappe.local.response.http_status_code = 403
		return {"status": "invalid signature"}

	try:
		payload = frappe.parse_json(raw_body.decode("utf-8"))
	except Exception:
		return {"status": "ignored"}

	for entry in payload.get("entry") or []:
		for change in entry.get("changes") or []:
			value = change.get("value") or {}
			for message in value.get("messages") or []:
				_process_message(message, settings)

	# Meta expects a fast 200 regardless of what happened inside (retries
	# on non-200/timeout) - real per-message errors are logged, not
	# surfaced here.
	return {"status": "ok"}


def _verify_signature(raw_body, settings):
	"""The actual security boundary for an endpoint that must be
	allow_guest=True: recompute the HMAC Meta signs every webhook POST
	with, using the App Secret, and constant-time compare against the
	X-Hub-Signature-256 header. Anything that fails this is discarded
	before any employee lookup or data access happens."""
	from frappe.utils.password import get_decrypted_password

	app_secret = get_decrypted_password("HR Chatbot Settings", "HR Chatbot Settings", "whatsapp_app_secret", raise_exception=False)
	if not app_secret:
		return False

	header = frappe.get_request_header("X-Hub-Signature-256") or ""
	if not header.startswith("sha256="):
		return False
	expected = hmac.new(app_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
	return hmac.compare_digest(expected, header[len("sha256=") :])


def _process_message(message, settings):
	if message.get("type") != "text":
		_reply(message.get("from"), settings, "I can only read text messages right now - please type your question.")
		return

	message_id = message.get("id")
	if message_id and not _claim_message(message_id):
		return  # already handled - a Meta retry, not a new message

	sender = message.get("from")
	text = ((message.get("text") or {}).get("body") or "").strip()
	if not sender or not text:
		return

	try:
		reply_text, attachments = _answer_as_employee(sender, text)
	except Exception:
		frappe.log_error(title="WhatsApp chatbot message handling failed")
		reply_text, attachments = "Something went wrong on my end. Please try again in a moment.", []

	_reply(sender, settings, reply_text)
	for attachment in attachments:
		_send_document(sender, settings, attachment["content"], attachment["filename"])


def _claim_message(message_id):
	"""True the first time this message id is seen, False on a repeat -
	same short-lived-cache idempotency pattern as _remember_intent/
	_recall_intent in hr_chatbot.py, applied to Meta's own retries instead
	of a chat follow-up."""
	cache_key = f"whatsapp_msg:{message_id}"
	if frappe.cache().get_value(cache_key, expires=True):
		return False
	frappe.cache().set_value(cache_key, 1, expires_in_sec=_DEDUPE_TTL_SECONDS)
	return True


def _answer_as_employee(sender_wa_id, message_text):
	"""Returns (text, attachments) - attachments is a list of {"filename",
	"content"} dicts for any downloadable document (currently: payslip
	PDFs) the reply pointed at. PDF bytes are generated here, inside the
	frappe.set_user(employee.user_id) block, deliberately - the desk-side
	download_pdf link ask() embeds requires a logged-in desk session and
	is useless to a WhatsApp user on their phone, so instead of sending
	that raw internal URL we render the PDF ourselves, still permission-
	scoped to the resolved employee exactly like every other lookup, and
	send the actual file."""
	employee = _resolve_employee_by_phone(sender_wa_id)
	if employee is None:
		return (
			"I couldn't find an employee record linked to this WhatsApp number. "
			"Please ask HR to add/update your mobile number on file, then message me again."
		), []
	if employee is _AMBIGUOUS:
		return (
			"This WhatsApp number matches more than one employee record on file, "
			"so I can't safely tell who's asking. Please ask HR to fix the duplicate "
			"mobile number, then message me again."
		), []
	if not employee.user_id:
		return (
			"Your employee record isn't linked to a login account yet, so I can't "
			"look anything up for you here. Please ask HR to link one."
		), []

	from arjun_chatbot.api.hr_chatbot import ask

	original_user = frappe.session.user
	attachments = []
	try:
		frappe.set_user(employee.user_id)
		reply = ask(message_text)
		reply_html, pdf_targets = _extract_pdf_targets(reply.get("reply") or "")
		for doctype, name in pdf_targets:
			try:
				pdf_bytes = frappe.get_print(doctype, name, as_pdf=True)
				filename = f"{name}.pdf".replace("/", "-")
				attachments.append({"filename": filename, "content": pdf_bytes})
			except Exception:
				frappe.log_error(title="WhatsApp chatbot PDF generation failed")
	finally:
		frappe.set_user(original_user)

	return _html_to_whatsapp_text(reply_html), attachments


_AMBIGUOUS = object()


def _normalize_phone_last10(value):
	digits = re.sub(r"\D", "", value or "")
	return digits[-10:] if len(digits) >= 10 else digits


def _resolve_employee_by_phone(wa_id):
	"""Match the sender's WhatsApp number against Employee.cell_number on
	file, comparing only the last 10 digits so +91/leading-zero/spacing
	differences in how a number was originally typed in don't cause a
	false miss. Returns an Employee _dict, None (no match), or the
	_AMBIGUOUS sentinel (more than one match) - callers must never guess
	which employee on an ambiguous match."""
	last10 = _normalize_phone_last10(wa_id)
	if len(last10) < 10:
		return None

	rows = frappe.db.sql(
		"""
		select name, user_id, cell_number
		from `tabEmployee`
		where status = 'Active'
			and cell_number is not null and cell_number != ''
			and right(regexp_replace(cell_number, '[^0-9]', ''), 10) = %(last10)s
		""",
		{"last10": last10},
		as_dict=True,
	)
	if len(rows) == 1:
		return rows[0]
	if len(rows) > 1:
		return _AMBIGUOUS
	return None


_BR_RE = re.compile(r"<br\s*/?>", re.I)
_LINK_RE = re.compile(r"<a\s+href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>", re.I | re.S)
# The click-to-reveal markup from hr_chatbot._augment_masked_reveals -
# WhatsApp has no click interaction, and a personal phone's chat backup
# (Google Drive/iCloud) is a real exposure path, so masked values stay
# masked-only over this channel by design: keep the masked text, drop the
# button (and its token) entirely rather than trying to make reveal work here.
_MASKED_VALUE_RE = re.compile(r"<span class=['\"]hrbot-masked-value['\"]>(.*?)</span>\s*<button[^>]*>.*?</button>", re.I | re.S)
_ANY_TAG_RE = re.compile(r"<[^>]+>")


# Matches the download_pdf links ai_query.py's get_document_pdf_url and
# hr_chatbot.py's _payslip build (see both) - extracted and handled
# separately from _LINK_RE so the doctype/name can be pulled out and a
# real file generated, instead of leaving a desk-session-only URL in the
# text (see _answer_as_employee).
_PDF_LINK_RE = re.compile(
	r"<a\s+href=['\"]/api/method/frappe\.utils\.print_format\.download_pdf\?doctype=([^&'\"]+)&(?:amp;)?name=([^'\"]+)['\"][^>]*>.*?</a>",
	re.I | re.S,
)


def _extract_pdf_targets(reply_html):
	from urllib.parse import unquote

	targets = []

	def _replace(match):
		targets.append((unquote(match.group(1)), unquote(match.group(2))))
		return "(see the attached PDF)"

	return _PDF_LINK_RE.sub(_replace, reply_html), targets


def _absolute_url(url):
	"""ask()'s replies embed site-relative links ('/app/...') that only
	make sense inside the desk widget's own page. WhatsApp shows plain
	text, so a bare relative path is meaningless - prefix it with the
	real host this exact webhook request just arrived on (frappe.request
	is the live incoming request here, so this is always correct even
	as a dev tunnel's URL changes between sessions, unlike a hardcoded
	site URL would be)."""
	if url.startswith("http://") or url.startswith("https://"):
		return url
	request = getattr(frappe.local, "request", None)
	base = request.url_root.rstrip("/") if request else frappe.utils.get_url()
	# Tunnels (ngrok etc.) terminate TLS and forward plain HTTP internally,
	# so request.url_root reports http:// even though the real public URL
	# - the one WhatsApp/the recipient actually needs - is https://.
	if base.startswith("http://"):
		base = "https://" + base[len("http://") :]
	return base + url


def _html_to_whatsapp_text(reply_html):
	text = _MASKED_VALUE_RE.sub(r"\1", reply_html)
	text = _LINK_RE.sub(lambda m: f"{m.group(2)}: {_absolute_url(m.group(1))}", text)
	text = _BR_RE.sub("\n", text)
	# Safety net, not the primary path: strips any other stray HTML (e.g.
	# a tag an AI-generated reply emitted despite instructions not to) so
	# a person never sees raw markup in their WhatsApp chat.
	text = _ANY_TAG_RE.sub("", text)
	return html.unescape(text).strip()


def _credentials(settings):
	from frappe.utils.password import get_decrypted_password

	access_token = get_decrypted_password("HR Chatbot Settings", "HR Chatbot Settings", "whatsapp_access_token", raise_exception=False)
	phone_number_id = settings.get("whatsapp_phone_number_id")
	if not access_token or not phone_number_id:
		frappe.log_error(title="WhatsApp chatbot not fully configured", message="Missing access token or phone number ID")
		return None, None
	return access_token, phone_number_id


def _reply(to, settings, text):
	if not to:
		return
	access_token, phone_number_id = _credentials(settings)
	if not access_token:
		return

	import requests

	try:
		requests.post(
			f"https://graph.facebook.com/{_GRAPH_API_VERSION}/{phone_number_id}/messages",
			headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
			json={
				"messaging_product": "whatsapp",
				"to": to,
				"type": "text",
				# WhatsApp's own hard cap on a single text message's length.
				"text": {"body": text[:4096]},
			},
			timeout=_SEND_TIMEOUT,
		)
	except Exception:
		frappe.log_error(title="WhatsApp chatbot send failed")


def _send_document(to, settings, pdf_bytes, filename):
	"""Uploads the PDF to Meta's Media API and sends it as a native
	WhatsApp document attachment - not a link, so it works without the
	recipient needing any desk login/session at all."""
	if not to:
		return
	access_token, phone_number_id = _credentials(settings)
	if not access_token:
		return

	import requests

	try:
		upload = requests.post(
			f"https://graph.facebook.com/{_GRAPH_API_VERSION}/{phone_number_id}/media",
			headers={"Authorization": f"Bearer {access_token}"},
			data={"messaging_product": "whatsapp", "type": "application/pdf"},
			files={"file": (filename, pdf_bytes, "application/pdf")},
			timeout=_SEND_TIMEOUT,
		)
		upload.raise_for_status()
		media_id = upload.json()["id"]

		requests.post(
			f"https://graph.facebook.com/{_GRAPH_API_VERSION}/{phone_number_id}/messages",
			headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
			json={
				"messaging_product": "whatsapp",
				"to": to,
				"type": "document",
				"document": {"id": media_id, "filename": filename},
			},
			timeout=_SEND_TIMEOUT,
		).raise_for_status()
	except Exception:
		frappe.log_error(title="WhatsApp chatbot document send failed")
