"""Minimal browser UI for Maintenance Copilot."""

from __future__ import annotations


def render_index_html() -> str:
    """Render a lightweight demo UI without extra frontend tooling."""

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Maintenance Copilot</title>
  <style>
    :root { color-scheme: light; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; }
    body { margin: 0; background: linear-gradient(180deg, #f6f8fc, #eef2f7); color: #111827; }
    .wrap { max-width: 980px; margin: 0 auto; padding: 32px 20px 64px; }
    .card { background: white; border: 1px solid #dbe3ee; border-radius: 8px; box-shadow: 0 20px 60px rgba(15, 23, 42, .08); padding: 24px; }
    h1 { margin: 0 0 8px; font-size: 2rem; }
    p { color: #475569; }
    label { display: block; font-weight: 600; margin-top: 16px; margin-bottom: 8px; }
    input, textarea { width: 100%; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 14px; font: inherit; }
    textarea { min-height: 120px; resize: vertical; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    button { margin-top: 18px; background: #0f172a; color: white; border: 0; border-radius: 8px; padding: 12px 18px; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: .6; cursor: wait; }
    button:hover { background: #1e293b; }
    pre { white-space: pre-wrap; background: #0f172a; color: #e2e8f0; padding: 16px; border-radius: 12px; overflow-x: auto; }
    .meta { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 18px; }
    .pill { background: #e2e8f0; border-radius: 999px; padding: 6px 10px; display: inline-block; margin-right: 6px; margin-bottom: 6px; }
    @media (max-width: 720px) { .row, .meta { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Maintenance Copilot</h1>
      <p>Ask about fault logs, production lines, and machine issues. The UI sends your API key as <code>X-API-Key</code>.</p>
      <div class="row">
        <div>
          <label for="apiKey">API Key</label>
          <input id="apiKey" type="password" placeholder="paste your API key" />
        </div>
        <div>
          <label for="threadId">Thread ID</label>
          <input id="threadId" type="text" placeholder="optional conversation thread id" />
        </div>
      </div>
      <label for="question">Question</label>
      <textarea id="question" placeholder="e.g. Line 3'te motor failure neden arttı?"></textarea>
      <button id="askBtn">Ask</button>
      <div class="meta">
        <div><strong>Query Type</strong><div id="queryType">-</div></div>
        <div><strong>Confidence</strong><div id="confidence">-</div></div>
        <div><strong>Request ID</strong><div id="requestId">-</div></div>
        <div><strong>Thread</strong><div id="threadOut">-</div></div>
      </div>
      <h3>Router</h3>
      <pre id="router">Waiting for a question...</pre>
      <h3>Evaluation</h3>
      <pre id="evaluation">Waiting for a question...</pre>
      <h3>Answer</h3>
      <pre id="answer">Waiting for a question...</pre>
      <h3>Citations</h3>
      <div id="citations"></div>
    </div>
  </div>
  <script>
    const apiKeyEl = document.getElementById('apiKey');
    const threadIdEl = document.getElementById('threadId');
    const questionEl = document.getElementById('question');
    const answerEl = document.getElementById('answer');
    const citationsEl = document.getElementById('citations');
    const queryTypeEl = document.getElementById('queryType');
    const confidenceEl = document.getElementById('confidence');
    const requestIdEl = document.getElementById('requestId');
    const threadOutEl = document.getElementById('threadOut');
    const routerEl = document.getElementById('router');
    const evaluationEl = document.getElementById('evaluation');
    const askBtn = document.getElementById('askBtn');

    function renderPill(text) {
      const node = document.createElement('div');
      node.className = 'pill';
      node.textContent = text;
      return node;
    }

    apiKeyEl.value = localStorage.getItem('maintenance-api-key') || '';

    askBtn.addEventListener('click', async () => {
      localStorage.setItem('maintenance-api-key', apiKeyEl.value);
      const payload = { question: questionEl.value };
      if (threadIdEl.value.trim()) payload.thread_id = threadIdEl.value.trim();

      answerEl.textContent = 'Loading...';
      citationsEl.innerHTML = '';
      askBtn.disabled = true;

      try {
        const response = await fetch('/ask', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': apiKeyEl.value,
          },
          body: JSON.stringify(payload),
        });

        requestIdEl.textContent = response.headers.get('X-Request-ID') || '-';
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Request failed');
        queryTypeEl.textContent = data.query_type || '-';
        confidenceEl.textContent = typeof data.router_confidence === 'number'
          ? data.router_confidence.toFixed(2)
          : '-';
        threadOutEl.textContent = data.thread_id || '-';
        threadIdEl.value = data.thread_id || threadIdEl.value;
        routerEl.textContent = JSON.stringify({
          reasoning: data.router_reasoning || '',
          filters: data.filters || {},
        }, null, 2);
        evaluationEl.textContent = JSON.stringify({
          is_sufficient: data.is_sufficient,
          answer_retry_count: data.retry_count,
          evidence_retry_count: data.evidence_retry_count,
          retry_target: data.retry_target,
          retry_targets: data.retry_targets || [],
          hit_retry_cap: data.hit_retry_cap,
          missing_aspects: data.missing_aspects || [],
          reasoning: data.evaluation_reasoning || '',
        }, null, 2);
        answerEl.textContent = data.answer || 'No answer returned';
        const citations = data.citations || [];
        citationsEl.innerHTML = '';
        if (citations.length) {
          citations.forEach(item => citationsEl.appendChild(renderPill(item)));
        } else {
          citationsEl.textContent = 'No citations returned.';
        }
      } catch (error) {
        answerEl.textContent = error.message || 'Request failed';
        routerEl.textContent = 'Request failed';
        evaluationEl.textContent = 'Request failed';
      } finally {
        askBtn.disabled = false;
      }
    });
  </script>
</body>
</html>"""
