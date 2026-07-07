const $ = (selector) => document.querySelector(selector);

const apiBaseInput = $("#apiBase");
const healthOutput = $("#healthOutput");
const askOutput = $("#askOutput");
const searchOutput = $("#searchOutput");

function apiBase() {
  return apiBaseInput.value.replace(/\/$/, "");
}

function render(target, value) {
  target.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

async function request(path, options = {}) {
  const response = await fetch(`${apiBase()}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });

  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json") ? await response.json() : await response.text();

  if (!response.ok) {
    const detail = typeof body === "object" ? body.detail || JSON.stringify(body) : body;
    throw new Error(`${response.status} ${response.statusText}: ${detail}`);
  }

  return body;
}

$("#healthBtn").addEventListener("click", async () => {
  render(healthOutput, "Checking...");
  try {
    render(healthOutput, await request("/health"));
  } catch (error) {
    render(healthOutput, error.message);
  }
});

$("#askForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  render(askOutput, "Asking...");
  try {
    render(
      askOutput,
      await request("/ask", {
        method: "POST",
        body: JSON.stringify({ question: $("#question").value }),
      }),
    );
  } catch (error) {
    render(askOutput, error.message);
  }
});

$("#searchForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  render(searchOutput, "Searching...");
  try {
    render(
      searchOutput,
      await request("/search", {
        method: "POST",
        body: JSON.stringify({
          query: $("#query").value,
          top_k: Number($("#topK").value),
          include_text: $("#includeText").checked,
        }),
      }),
    );
  } catch (error) {
    render(searchOutput, error.message);
  }
});
