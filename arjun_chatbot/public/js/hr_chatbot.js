// arjun_chatbot/public/js/hr_chatbot.js
//
// Floating "ask HR" widget on every Desk page. Pure data lookup, no LLM -
// every reply comes straight from arjun_chatbot.api.hr_chatbot.ask(), which
// only ever answers with the logged-in user's own real HRMS data (see that
// file for the full list of things it understands). Guest never sees this
// (app_include_js is desk-only, never loaded on the onboarding portal).
(function () {
	if (frappe.session.user === "Guest") return;

	function escapeHtml(s) {
		var div = document.createElement("div");
		div.textContent = s;
		return div.innerHTML;
	}

	function addMessage(text, who) {
		var messages = document.getElementById("hrbot-messages");
		var div = document.createElement("div");
		div.className = "hrbot-msg " + who;
		// Bot replies are built entirely server-side from fixed templates
		// (see hr_chatbot.py) - safe to render as HTML. User's own typed
		// text is escaped since it's echoed back into the transcript.
		div.innerHTML = who === "bot" ? text : escapeHtml(text);
		messages.appendChild(div);
		messages.scrollTop = messages.scrollHeight;
	}

	function ask(message) {
		addMessage(message, "user");
		frappe.call({
			method: "arjun_chatbot.api.hr_chatbot.ask",
			args: { message: message },
			callback: function (r) {
				addMessage((r.message && r.message.reply) || "Sorry, something went wrong.", "bot");
			},
			error: function () {
				addMessage("Sorry, something went wrong. Please try again.", "bot");
			},
		});
	}

	// Drag the whole widget by the chat icon - the panel isn't independently
	// draggable, it just always sits anchored to wherever the icon currently
	// is (recalculated from the icon's live position rather than tracked
	// separately, since a hidden/closed panel's own position can't be read
	// via getBoundingClientRect - display:none elements report a zero rect).
	// Icon position persists across page loads via localStorage.
	var POSITION_KEY = "hrbot_toggle_position";
	var PANEL_GAP = 10;
	var DRAG_THRESHOLD = 5;

	function positionPanel(toggle, panel) {
		var t = toggle.getBoundingClientRect();
		// offsetWidth/Height read 0 while display:none (before first open) -
		// fall back to the CSS defaults in that case.
		var panelW = panel.offsetWidth || 320;
		var panelH = panel.offsetHeight || 420;
		var left = Math.max(4, Math.min(window.innerWidth - panelW - 4, t.right - panelW));
		var top = Math.max(4, Math.min(window.innerHeight - panelH - 4, t.top - panelH - PANEL_GAP));
		panel.style.left = left + "px";
		panel.style.top = top + "px";
		panel.style.right = "auto";
		panel.style.bottom = "auto";
	}

	function restoreTogglePosition(toggle) {
		try {
			var saved = JSON.parse(localStorage.getItem(POSITION_KEY) || "null");
			if (saved && saved.left && saved.top) {
				toggle.style.left = saved.left;
				toggle.style.top = saved.top;
				toggle.style.right = "auto";
				toggle.style.bottom = "auto";
			}
		} catch (e) {
			/* ignore a corrupted/unavailable stored position - falls back to the CSS default */
		}
	}

	// Returns a wasDragged() check so the click handler that opens/closes
	// the panel can tell a real drag apart from a plain click and skip
	// toggling right after someone finishes dragging it.
	function makeDraggable(toggle, panel) {
		var dragging = false, moved = false, startX, startY, startLeft, startTop;

		function down(e) {
			dragging = true;
			moved = false;
			var p = e.touches ? e.touches[0] : e;
			var rect = toggle.getBoundingClientRect();
			startX = p.clientX;
			startY = p.clientY;
			startLeft = rect.left;
			startTop = rect.top;
			e.preventDefault();
		}

		function move(e) {
			if (!dragging) return;
			var p = e.touches ? e.touches[0] : e;
			var dx = p.clientX - startX;
			var dy = p.clientY - startY;
			if (Math.abs(dx) > DRAG_THRESHOLD || Math.abs(dy) > DRAG_THRESHOLD) moved = true;
			if (!moved) return;
			var left = Math.max(4, Math.min(window.innerWidth - toggle.offsetWidth - 4, startLeft + dx));
			var top = Math.max(4, Math.min(window.innerHeight - toggle.offsetHeight - 4, startTop + dy));
			toggle.style.left = left + "px";
			toggle.style.top = top + "px";
			toggle.style.right = "auto";
			toggle.style.bottom = "auto";
			positionPanel(toggle, panel);
		}

		function up() {
			if (!dragging) return;
			dragging = false;
			if (moved) {
				try {
					localStorage.setItem(POSITION_KEY, JSON.stringify({ left: toggle.style.left, top: toggle.style.top }));
				} catch (e) {
					/* localStorage unavailable (private browsing etc) - dragging still works, just doesn't persist */
				}
			}
		}

		toggle.addEventListener("mousedown", down);
		toggle.addEventListener("touchstart", down, { passive: false });
		document.addEventListener("mousemove", move);
		document.addEventListener("touchmove", move, { passive: false });
		document.addEventListener("mouseup", up);
		document.addEventListener("touchend", up);

		return function wasDragged() {
			return moved;
		};
	}

	function buildWidget() {
		var toggle = document.createElement("div");
		toggle.id = "hrbot-toggle";
		toggle.title = "Ask HR";
		toggle.innerHTML = "💬";
		document.body.appendChild(toggle);

		var panel = document.createElement("div");
		panel.id = "hrbot-panel";
		panel.innerHTML =
			"<div id='hrbot-header'><span>Ask HR</span><span class='hrbot-close'>&times;</span></div>" +
			"<div id='hrbot-messages'></div>" +
			"<div id='hrbot-input-row'>" +
			"<input id='hrbot-input' type='text' placeholder='e.g. What's my leave balance?'>" +
			"<button id='hrbot-send' type='button'>Send</button>" +
			"</div>";
		document.body.appendChild(panel);

		restoreTogglePosition(toggle);
		positionPanel(toggle, panel);
		var wasDragged = makeDraggable(toggle, panel);

		var opened = false;
		toggle.addEventListener("click", function () {
			// A drag ends with a mouseup, which the browser follows with a
			// click on the same element - without this check, finishing a
			// drag would also toggle the panel open/closed unintentionally.
			if (wasDragged()) return;
			positionPanel(toggle, panel);
			panel.classList.toggle("open");
			if (!opened) {
				opened = true;
				addMessage(
					"Hi! I can look up your leave balance, leave status, attendance, payslip, holidays, or reporting manager. What would you like to know?",
					"bot"
				);
			}
		});
		panel.querySelector(".hrbot-close").addEventListener("click", function () {
			panel.classList.remove("open");
		});

		var input = document.getElementById("hrbot-input");
		function send() {
			var value = input.value.trim();
			if (!value) return;
			input.value = "";
			ask(value);
		}
		document.getElementById("hrbot-send").addEventListener("click", send);
		input.addEventListener("keydown", function (e) {
			if (e.key === "Enter") send();
		});
	}

	if (document.readyState === "loading") {
		document.addEventListener("DOMContentLoaded", buildWidget);
	} else {
		buildWidget();
	}
})();
