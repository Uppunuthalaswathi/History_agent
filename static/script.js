(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const messageInput = $("messageInput"), sendBtn = $("sendBtn"), chatMessages = $("chatMessages"), resetBtn = $("resetBtn"), messageForm = $("messageForm");
  const uploadForm = $("uploadForm"), fileInput = $("fileInput"), fileName = $("fileName"), uploadButton = $("uploadButton"), uploadStatus = $("uploadStatus"), uploadedDocument = $("uploadedDocument"), uploadedDocumentName = $("uploadedDocumentName"), attachBtn = $("attachBtn");
  const allowed = [".pdf", ".txt", ".docx"], maxBytes = 20 * 1024 * 1024;
  let uploading = false;
  const selectedFile = () => fileInput.files?.[0] || null;
  function setStatus(message = "", type = "") { uploadStatus.textContent = message; uploadStatus.className = `upload-status ${type}`; }
  function updateFileSelection() {
    const file = selectedFile();
    if (!file) { fileName.textContent = "No file selected"; uploadButton.disabled = true; return; }
    fileName.textContent = file.name;
    const extension = `.${file.name.split(".").pop().toLowerCase()}`;
    if (!allowed.includes(extension)) { setStatus("❌ Unsupported file type.", "error"); uploadButton.disabled = true; return; }
    if (file.size > maxBytes) { setStatus("❌ File is too large. Maximum size is 20 MB.", "error"); uploadButton.disabled = true; return; }
    setStatus(); uploadButton.disabled = false;
  }
  fileInput.addEventListener("change", updateFileSelection);
  attachBtn.addEventListener("click", () => fileInput.click());
  function scrollBottom() { chatMessages.scrollTop = chatMessages.scrollHeight; }
  function markdown(text) {
    const raw = String(text || "");
    if (window.marked && window.DOMPurify) return DOMPurify.sanitize(marked.parse(raw, { gfm: true, breaks: true }), { USE_PROFILES: { html: true }, ADD_ATTR: ["target"] });
    const div = document.createElement("div"); div.textContent = raw; return div.innerHTML.replace(/\n/g, "<br>");
  }
  function addMessage(text, sender, isHtml = false) {
    document.querySelector(".welcome-message")?.remove();
    const article = document.createElement("article"); article.className = `message ${sender}`; article.setAttribute("aria-label", `${sender === "user" ? "User" : "AI"} message`);
    const avatar = document.createElement("div"); avatar.className = "message-avatar"; avatar.setAttribute("aria-hidden", "true"); avatar.textContent = sender === "user" ? "👤" : "🤖";
    const content = document.createElement("div"); content.className = "message-content"; content.innerHTML = isHtml ? DOMPurify.sanitize(text, { USE_PROFILES: { html: true }, ADD_ATTR: ["target"] }) : markdown(text);
    article.append(avatar, content); chatMessages.append(article); scrollBottom(); return article;
  }
  function systemMessage(text) { const item = document.createElement("div"); item.className = "system-message"; item.setAttribute("role", "status"); item.textContent = text; chatMessages.append(item); scrollBottom(); }
  function typing() { const item = addMessage("<div class='typing-indicator'><span></span><span></span><span></span></div>", "agent", true); item.id = "typing-indicator"; return item; }
  function chatBusy(value) { messageInput.disabled = value; sendBtn.disabled = value; }
  async function sendMessage(event) {
    event.preventDefault(); const message = messageInput.value.trim(); if (!message) return;
    addMessage(message, "user"); messageInput.value = ""; chatBusy(true); const indicator = typing();
    try {
      const response = await fetch("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message }) });
      const data = await response.json().catch(() => ({})); if (!response.ok || data.error) throw new Error(data.error || "Unable to get a response.");
      addMessage(data.response_html || data.response || "", "agent", Boolean(data.response_html));
    } catch (error) { addMessage(`Error: ${error.message}`, "agent"); }
    finally { indicator.remove(); chatBusy(false); messageInput.focus(); }
  }
  messageForm.addEventListener("submit", sendMessage);
  uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault(); const file = selectedFile();
    if (!file) { setStatus("❌ Please select a document first.", "error"); return; }
    if (uploading || uploadButton.disabled) return;
    uploading = true; uploadButton.disabled = true; uploadButton.textContent = "Uploading…"; setStatus("Uploading and indexing your document…", "loading");
    try {
      const formData = new FormData(); formData.append("file", file);
      const response = await fetch("/upload", { method: "POST", body: formData });
      const data = await response.json().catch(() => ({})); if (!response.ok || !data.success) throw new Error(data.error || "Upload failed.");
      const name = data.filename || file.name, chunks = Number.isFinite(data.chunks) ? ` (${data.chunks} chunks)` : "";
      setStatus(`✓ Document indexed successfully${chunks}`, "success"); uploadedDocument.hidden = false; uploadedDocumentName.textContent = name;
      systemMessage(`📚 Knowledge base updated with ${name}. You can now ask questions about this document.`);
      fileInput.value = ""; fileName.textContent = "No file selected";
    } catch (error) { setStatus(`❌ ${error.message || "Upload failed."}`, "error"); }
    finally { uploading = false; uploadButton.textContent = "Upload & Index"; uploadButton.disabled = !selectedFile(); }
  });
  resetBtn.addEventListener("click", async () => {
    resetBtn.disabled = true;
    try { const response = await fetch("/reset", { method: "POST" }); if (!response.ok) throw new Error(); chatMessages.innerHTML = '<div class="welcome-message"><p>👋 <strong>Let\'s chat about computing history!</strong></p><p>You can also upload your own <strong>PDF, TXT, or DOCX</strong> document above and ask questions about it.</p></div>'; }
    catch { systemMessage("Unable to reset the conversation. Please try again."); }
    finally { resetBtn.disabled = false; messageInput.focus(); }
  });
  messageInput.focus();
})();
