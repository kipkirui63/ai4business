const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const input = document.getElementById("message-input");
const resetButton = document.getElementById("reset-button");
const promptButtons = document.querySelectorAll(".prompt");

let sessionId = null;

function appendMessage(role, text) {
    const wrapper = document.createElement("article");
    wrapper.className = `message ${role}`;

    const label = document.createElement("span");
    label.className = "label";
    label.textContent = role === "bot" ? "FreightBot" : "You";

    const content = document.createElement("div");
    content.textContent = text;

    wrapper.appendChild(label);
    wrapper.appendChild(content);
    chatLog.appendChild(wrapper);
    chatLog.scrollTop = chatLog.scrollHeight;
}

async function sendMessage(message) {
    appendMessage("user", message);

    const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            message,
            session_id: sessionId,
        }),
    });

    if (!response.ok) {
        appendMessage("bot", "The service could not process that request.");
        return;
    }

    const data = await response.json();
    sessionId = data.session_id;
    appendMessage("bot", data.reply);
}

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) {
        return;
    }

    input.value = "";
    await sendMessage(message);
    input.focus();
});

input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        chatForm.requestSubmit();
    }
});

resetButton.addEventListener("click", async () => {
    chatLog.innerHTML = "";

    const response = await fetch("/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: "" }),
    });

    if (response.ok) {
        const data = await response.json();
        sessionId = data.session_id;
    }

    appendMessage(
        "bot",
        "Hello, this is FreightBot. I can help with tracking, quotes, pickup booking, delays, and damaged freight."
    );
    input.focus();
});

for (const button of promptButtons) {
    button.addEventListener("click", () => {
        input.value = button.dataset.prompt;
        input.focus();
    });
}

appendMessage(
    "bot",
    "Hello, this is FreightBot. I can help with tracking, quotes, pickup booking, delays, and damaged freight."
);
