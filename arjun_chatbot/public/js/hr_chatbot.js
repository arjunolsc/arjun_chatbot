// arjun_chatbot/public/js/hr_chatbot.js
//
// ALIA - ATLAS Intelligent Assistant. Floating chat widget on every Desk
// page. Pure data lookup, no LLM by default - every reply comes straight
// from arjun_chatbot.api.hr_chatbot.ask(), which only ever answers with
// the logged-in user's own real HRMS data (see that file for the full
// list of things it understands). Guest never sees this (app_include_js
// is desk-only, never loaded on the onboarding portal).
(function () {
	if (frappe.session.user === "Guest") return;

	// ALIA's avatar - a provided illustration (public/images/alia_avatar.png),
	// cropped to a square headshot and resized down from the original
	// 800x1300/1.9MB source to 300x300 so it stays fast to load as a small
	// icon repeated across every desk page.
	var ALIA_AVATAR_HTML =
		"<img src='/assets/arjun_chatbot/images/alia_avatar.png' alt='ALIA' class='hrbot-avatar-img'>";

	var SEND_ICON_SVG =
		"<svg viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'>" +
		"<path d='M3 11l17-8-6 17-3-7-8-2z' fill='currentColor'/>" +
		"</svg>";

	var QUICK_REPLIES = [
		{ label: "💰 Payslip", query: "my latest payslip" },
		{ label: "🌴 Leave Balance", query: "what's my leave balance" },
		{ label: "📅 Attendance", query: "my attendance this month" },
		{ label: "🙍 My Profile", query: "my profile" },
	];

	function escapeHtml(s) {
		var div = document.createElement("div");
		div.textContent = s;
		return div.innerHTML;
	}

	function addMessage(text, who) {
		var messages = document.getElementById("hrbot-messages");
		var row = document.createElement("div");
		row.className = "hrbot-row " + who;
		if (who === "bot") {
			var avatar = document.createElement("div");
			avatar.className = "hrbot-row-avatar";
			avatar.innerHTML = ALIA_AVATAR_HTML;
			row.appendChild(avatar);
		}
		var bubble = document.createElement("div");
		bubble.className = "hrbot-msg " + who;
		// Bot replies are built entirely server-side from fixed templates
		// (see hr_chatbot.py) - safe to render as HTML. User's own typed
		// text is escaped since it's echoed back into the transcript.
		bubble.innerHTML = who === "bot" ? text : escapeHtml(text);
		row.appendChild(bubble);
		messages.appendChild(row);
		messages.scrollTop = messages.scrollHeight;
		return row;
	}

	function addChips() {
		var messages = document.getElementById("hrbot-messages");
		var wrap = document.createElement("div");
		wrap.className = "hrbot-chips";
		QUICK_REPLIES.forEach(function (item) {
			var chip = document.createElement("button");
			chip.type = "button";
			chip.className = "hrbot-chip";
			chip.textContent = item.label;
			chip.addEventListener("click", function () {
				ask(item.query);
			});
			wrap.appendChild(chip);
		});
		messages.appendChild(wrap);
	}

	function showTyping() {
		var messages = document.getElementById("hrbot-messages");
		var row = document.createElement("div");
		row.className = "hrbot-row bot";
		row.id = "hrbot-typing-row";
		var avatar = document.createElement("div");
		avatar.className = "hrbot-row-avatar";
		avatar.innerHTML = ALIA_AVATAR_HTML;
		row.appendChild(avatar);
		var bubble = document.createElement("div");
		bubble.className = "hrbot-msg bot hrbot-typing";
		bubble.innerHTML = "<span></span><span></span><span></span>";
		row.appendChild(bubble);
		messages.appendChild(row);
		messages.scrollTop = messages.scrollHeight;
	}

	function hideTyping() {
		var row = document.getElementById("hrbot-typing-row");
		if (row) row.remove();
	}

	function ask(message) {
		addMessage(message, "user");
		showTyping();
		frappe.call({
			method: "arjun_chatbot.api.hr_chatbot.ask",
			args: { message: message },
			callback: function (r) {
				hideTyping();
				addMessage((r.message && r.message.reply) || "Sorry, something went wrong.", "bot");
			},
			error: function () {
				hideTyping();
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
		var panelW = panel.offsetWidth || 336;
		var panelH = panel.offsetHeight || 460;
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
		toggle.title = "Ask ALIA";
		toggle.innerHTML = ALIA_AVATAR_HTML + "<span class='hrbot-dot'></span>";
		document.body.appendChild(toggle);

		var panel = document.createElement("div");
		panel.id = "hrbot-panel";
		panel.innerHTML =
			"<div id='hrbot-header'>" +
			"<div class='hrbot-header-left'>" +
			"<div class='hrbot-avatar'>" + ALIA_AVATAR_HTML + "</div>" +
			"<div><div class='hrbot-title'>ALIA</div>" +
			"<div class='hrbot-subtitle'><span class='hrbot-online-dot'></span>ATLAS Intelligent Assistant</div></div>" +
			"</div>" +
			"<span class='hrbot-close'>&times;</span>" +
			"</div>" +
			"<div id='hrbot-messages'></div>" +
			"<div id='hrbot-input-row'>" +
			"<input id='hrbot-input' type='text' placeholder='Ask ALIA anything...'>" +
			"<button id='hrbot-send' type='button'>" + SEND_ICON_SVG + "</button>" +
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
					"Hi, I'm ALIA 👋 Your ATLAS Intelligent Assistant. I can look up your leave, attendance, payslip, profile and a lot more - or just ask away!",
					"bot"
				);
				addChips();
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
