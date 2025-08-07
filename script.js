// 🌐 Chat logic
document.getElementById("chat-form").addEventListener("submit", async function (e) {
    e.preventDefault();

    const inputField = document.getElementById("user-input");
    const message = inputField.value.trim();
    if (!message) return;

    appendMessage("user", message);
    inputField.value = "";

    try {
        const response = await fetch("/get", {
            method: "POST",
            body: JSON.stringify({ message }),
            headers: {
                "Content-Type": "application/json"
            }
        });

        const data = await response.json();
        appendMessage("bot", data.reply);
    } catch (error) {
        appendMessage("bot", "⚠️ Sorry, something went wrong.");
        console.error("❌ Fetch error:", error);
    }
});

// 💬 Add message to chat box
function appendMessage(sender, text) {
    const chatBox = document.getElementById("chat-box");
    const msg = document.createElement("div");
    msg.classList.add("message", sender);
    msg.innerText = `${sender === "user" ? "You" : "Bot"}: ${text}`;
    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// 🌙 Dark Mode toggle
const toggle = document.getElementById("dark-mode-toggle");

toggle.addEventListener("change", () => {
    document.body.classList.toggle("dark-mode");
    localStorage.setItem("theme", document.body.classList.contains("dark-mode") ? "dark" : "light");
});

// 🚀 Load stored theme
window.addEventListener("DOMContentLoaded", () => {
    if (localStorage.getItem("theme") === "dark") {
        document.body.classList.add("dark-mode");
        toggle.checked = true;
    }
});
