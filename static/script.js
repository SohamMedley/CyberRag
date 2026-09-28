/* CYBER-RAG front end. Talks to the Flask API in app.py, no external deps. */
'use strict';

document.addEventListener('DOMContentLoaded', () => {
  const $ = (id) => document.getElementById(id);

  /* ------------------------------------------------------------------ state */
  let queue = [];          // files selected for upload: {file, base, stored, uploaded, failure}
  let status = null;       // last /status payload
  let busy = '';           // '', 'index', 'ask', 'refresh', 'delete'
  let view = 'library';
  let answerExists = false;
  let copyTimer;
  let pendingDelete = null;

  const VIEWS = {
    library: ['Library', 'Everything you know. Ready to explore.'],
    ask: ['Ask', 'A little curiosity goes a long way.'],
  };
  const MAX_BATCH_BYTES = 16 * 1024 * 1024;
  const HISTORY_TURNS = 3;
  let history = [];        // [{role, content}] sent with /ask for follow-ups

  /* ---------------------------------------------------------------- helpers */
  function node(tag, cls, text) {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text != null) el.textContent = text;
    return el;
  }

  function icon(name, cls = '') {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', cls);
    svg.setAttribute('aria-hidden', 'true');
    const use = document.createElementNS(svg.namespaceURI, 'use');
    use.setAttribute('href', '#' + name);
    svg.append(use);
    return svg;
  }

  function notify(title, text = '', kind = 'error') {
    $('noticeTitle').textContent = title;
    $('noticeText').textContent = text;
    $('notice').className = 'notice ' + kind;
    $('notice').hidden = false;
  }

  function clearNotice() {
    $('notice').hidden = true;
  }

  function switchView(next, scroll = true) {
    if (!VIEWS[next]) return;
    view = next;
    for (const key of Object.keys(VIEWS)) $(key + 'View').hidden = key !== next;
    $('viewTitle').textContent = VIEWS[next][0];
    $('viewSubtitle').textContent = VIEWS[next][1];
    document.title = `${VIEWS[next][0]} · CYBER-RAG`;
    document.querySelectorAll('[data-view]').forEach((button) => {
      const active = button.dataset.view === next;
      button.classList.toggle('selected', active);
      if (active) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
    });
    if (scroll) window.scrollTo({ top: 0, behavior: 'instant' });
  }

  function controls() {
    const locked = !!busy;
    for (const id of ['dropZone', 'fileInput', 'heroUpload', 'refreshButton']) $(id).disabled = locked;
    $('indexButton').disabled = locked || !status || !queue.length;
    $('askButton').disabled =
      locked || !status?.total_chunks_indexed || !status?.is_groq_key_set || !$('questionInput').value.trim();
    $('questionInput').readOnly = busy === 'ask';
    document.querySelectorAll('.queue-row button, .document-delete, [data-question]').forEach((button) => {
      button.disabled = locked;
    });
    $('composerHint').textContent =
      busy === 'ask' ? 'Generating an answer…'
      : busy === 'index' ? 'Processing documents. Keep this page open.'
      : busy === 'delete' ? 'Removing the document…'
      : busy === 'refresh' ? 'Refreshing…'
      : !status ? 'Server unavailable · refresh in Library'
      : !status.total_chunks_indexed ? 'Add and index documents in Library'
      : !status.is_groq_key_set ? 'Set GROQ_API_KEY in your .env, then restart the app'
      : `Ask across ${status.documents_count} document${status.documents_count === 1 ? '' : 's'}`;
    $('askContextText').textContent = !status
      ? 'Server unavailable'
      : `${status.documents_count} document${status.documents_count === 1 ? '' : 's'} in your library`;
    $('questionForm').setAttribute('aria-busy', String(busy === 'ask'));
    $('libraryView').setAttribute('aria-busy', String(busy === 'index'));
    resizeDock();
  }

  /* -------------------------------------------------------------------- api */
  async function api(path, options = {}, timeout = 15000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const res = await fetch(path, { ...options, signal: controller.signal, cache: 'no-store' });
      let data;
      try {
        data = await res.json();
      } catch {
        throw new Error(
          res.status === 413
            ? 'The whole upload request exceeds 16 MB. Remove a file or upload a smaller batch.'
            : `Unexpected server response (${res.status}). Check the server logs.`
        );
      }
      if (!res.ok) {
        const extra = [
          ...(data.details || []),
          ...(data.report || [])
            .filter((row) => row.status === 'failed')
            .map((row) => `${row.filename}: ${row.reason}`),
        ];
        throw new Error([data.error || `Request failed (${res.status}).`, ...extra].join('\n'));
      }
      return data;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('The server took too long to respond. It may still be working.');
      if (error instanceof TypeError) throw new Error('Could not reach the server. Check your connection and that the app is running.');
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  const post = (path, data, timeout) =>
    api(
      path,
      { method: 'POST', ...(data ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) } : {}) },
      timeout
    );

  /* ----------------------------------------------------------- file naming */
  function safeBase(name) {
    let base = name
      .normalize('NFKD')
      .replace(/[^\x00-\x7F]/g, '')
      .replace(/[\\/]/g, ' ')
      .split(/\s+/)
      .join('_')
      .replace(/[^A-Za-z0-9_.-]/g, '')
      .replace(/^[._]+|[._]+$/g, '');
    const dot = base.lastIndexOf('.');
    let stem = (dot > 0 ? base.slice(0, dot) : 'document').slice(0, 90) || 'document';
    if (/^(con|prn|aux|nul|com[0-9]|lpt[0-9])$/i.test(stem)) stem = '_' + stem;
    const ext = name.split('.').pop().toLowerCase();
    return stem + '.' + ext;
  }

  function displayName(name) {
    return name.replace(/__rag_[a-f0-9]{16}(?=\.(pdf|txt)$)/i, '');
  }

  function key(name) {
    return safeBase(displayName(name)).toLowerCase();
  }

  function storedName(name) {
    const bytes = new Uint8Array(8);
    crypto.getRandomValues(bytes);
    const token = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
    const dot = name.lastIndexOf('.');
    return name.slice(0, dot) + '__rag_' + token + name.slice(dot);
  }

  const formatBytes = (n) =>
    n >= 1048576 ? (n / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(n / 1024)) + ' KB';

  /* ----------------------------------------------------------------- upload */
  function renderQueue() {
    const list = $('queue');
    list.replaceChildren();
    $('queueHeading').hidden = !queue.length;
    $('queueHeading').querySelector('h4').textContent = 'Ready to upload';
    $('queueSize').textContent = formatBytes(queue.reduce((total, item) => total + item.file.size, 0));
    queue.forEach((item, index) => {
      const row = node('li', 'queue-row');
      const info = node('div');
      info.append(
        node('strong', '', item.file.name),
        node('small', '', item.failure || `${formatBytes(item.file.size)} · stored as ${item.base}`)
      );
      const remove = node('button', '', '×');
      remove.type = 'button';
      remove.setAttribute('aria-label', `Remove ${item.file.name}`);
      remove.onclick = () => {
        if (busy) return;
        queue.splice(index, 1);
        renderQueue();
      };
      row.append(info, remove);
      list.append(row);
    });
    controls();
  }

  function select(files) {
    if (busy) return;
    const problems = [];
    for (const file of files) {
      if (!/\.(pdf|txt)$/i.test(file.name)) {
        problems.push(`${file.name}: only PDF and TXT files are supported.`);
        continue;
      }
      if (!file.size) {
        problems.push(`${file.name}: the file is empty.`);
        continue;
      }
      const base = safeBase(file.name);
      const id = key(base);
      if (queue.some((item) => key(item.base) === id)) {
        problems.push(`${file.name}: a file with the same normalized name is already queued.`);
        continue;
      }
      if (status?.indexed_documents?.some((doc) => key(doc.filename) === id)) {
        problems.push(`${file.name}: already in your library. Delete it first if you want to replace it.`);
        continue;
      }
      if (queue.reduce((total, item) => total + item.file.size, 0) + file.size > MAX_BATCH_BYTES - 65536) {
        problems.push(`${file.name}: the batch would exceed 16 MB. Upload a smaller batch.`);
        continue;
      }
      queue.push({ file, base, stored: storedName(base), uploaded: false, failure: '' });
    }
    $('fileInput').value = '';
    renderQueue();
    if (problems.length) notify('Some files were not added', problems.join('\n'));
    else clearNotice();
  }

  async function indexFiles() {
    if (busy || !queue.length || !status) return;
    busy = 'index';
    controls();
    clearNotice();
    $('indexProgress').classList.add('processing');

    const reports = [];
    try {
      const fresh = await api('/status');
      status = fresh;
      renderStatus();
      if (queue.some((item) => fresh.indexed_documents.some((doc) => key(doc.filename) === key(item.base)))) {
        throw new Error('A queued file is already indexed. Remove it from the queue before continuing.');
      }

      const pending = queue.filter((item) => !item.uploaded);
      if (pending.length) {
        $('indexButton').firstElementChild.textContent = 'Uploading…';
        $('indexProgressText').textContent = 'Uploading your documents…';
        const form = new FormData();
        pending.forEach((item) => form.append('files', item.file, item.stored));
        const data = await api('/upload', { method: 'POST', body: form }, 120000);
        if (!Array.isArray(data.saved_files)) throw new Error('Upload response is missing saved filenames.');
        pending.forEach((item) => {
          item.uploaded = data.saved_files.includes(item.stored);
          if (!item.uploaded) item.failure = 'Not accepted by the server.';
        });
        reports.push(...(data.warnings || []));
      }

      const accepted = queue.filter((item) => item.uploaded);
      if (!accepted.length) throw new Error('No files were accepted by the server.');

      $('indexButton').firstElementChild.textContent = 'Indexing…';
      $('indexProgressText').textContent = 'Extracting, embedding and indexing…';
      const data = await post('/index', { filenames: accepted.map((item) => item.stored) }, 180000);

      const rows = Array.isArray(data.report) ? data.report : [];
      const done = new Set(
        rows
          .filter((row) => row.status === 'success' || row.status === 'skipped')
          .map((row) => row.filename)
      );
      queue = queue.filter((item) => !done.has(item.stored));
      rows
        .filter((row) => row.status === 'skipped')
        .forEach((row) => reports.push(`${displayName(row.filename)}: ${row.reason || 'already indexed.'}`));
      rows
        .filter((row) => row.status === 'failed' && queue.some((item) => item.stored === row.filename))
        .forEach((row) => {
          const item = queue.find((entry) => entry.stored === row.filename);
          item.failure = row.reason || 'Could not read the document.';
          reports.push(`${item.file.name}: ${item.failure}`);
        });

      $('indexProgressText').textContent = reports.length
        ? 'Some files need attention.'
        : 'Indexed successfully. Your library is ready.';
      notify(
        reports.length ? 'Some documents need attention' : 'Documents indexed',
        reports.length ? reports.join('\n') : `Added ${data.new_chunks_added} chunks. Switch to Ask to explore them.`,
        reports.length ? 'error' : 'success'
      );
      if (!reports.length) {
        queue = [];
        renderQueue();
      }
    } catch (error) {
      $('indexProgressText').textContent = 'Processing stopped. Check the message above.';
      notify('Could not index the documents', error.message);
    } finally {
      await refresh();
      busy = '';
      $('indexProgress').classList.remove('processing');
      $('indexButton').firstElementChild.textContent = 'Index documents';
      renderQueue();
    }
  }

  /* --------------------------------------------------------------- library */
  function renderStatus() {
    $('connection').classList.toggle('online', !!status);
    $('statusText').textContent = !status ? 'Offline' : status.is_groq_key_set ? 'Connected' : 'Key needed';
    $('docsCount').textContent = status ? status.documents_count : '—';
    $('chunksCount').textContent = status ? status.total_chunks_indexed : '—';

    const list = $('documentList');
    list.replaceChildren();

    if (!status || (status.indexed_documents || []).length === 0) {
      const empty = node('div', 'library-empty');
      empty.append(
        icon('doc'),
        node('strong', '', status ? 'Your library starts here' : 'Library unavailable'),
        node(
          'span',
          '',
          status
            ? 'Add a PDF or TXT file, then index it.\nYour documents will appear here.'
            : 'Check that the app is running, then select Refresh.'
        )
      );
      list.append(empty);
    }

    for (const doc of status?.indexed_documents || []) {
      const row = node('div', 'document');
      const isTxt = doc.filename.toLowerCase().endsWith('.txt');
      row.append(node('span', 'file-icon' + (isTxt ? ' txt' : ''), isTxt ? 'TXT' : 'PDF'));

      const info = node('div', 'document-info');
      info.append(
        node('strong', '', displayName(doc.filename)),
        node(
          'small',
          '',
          `${doc.pages} text section${doc.pages === 1 ? '' : 's'} · ${doc.chunks} chunk${doc.chunks === 1 ? '' : 's'}` +
            (doc.indexed_at ? ` · ${new Date(doc.indexed_at).toLocaleDateString()}` : '')
        )
      );
      info.title = 'Stored as: ' + doc.filename;
      row.append(info, icon('check', 'indexed-mark'));

      const remove = node('button', 'document-delete');
      remove.type = 'button';
      remove.append(icon('trash'));
      remove.setAttribute('aria-label', `Delete ${displayName(doc.filename)}`);
      remove.title = 'Delete this document from the index';
      remove.onclick = () => openDelete(displayName(doc.filename), doc.filename);
      row.append(remove);

      list.append(row);
    }
    controls();
  }

  async function refresh(announce = false) {
    try {
      const data = await api('/status');
      if (!Array.isArray(data.indexed_documents) || !Number.isFinite(data.total_chunks_indexed)) {
        throw new Error('Invalid status response. Check the GET /status endpoint.');
      }
      status = data;
      renderStatus();
      if (data.index_load_error) {
        notify(
          'The saved index could not be loaded',
          `${data.index_load_error}\nUpload and index your documents again to rebuild the library.`
        );
      }
      return true;
    } catch (error) {
      status = null;
      renderStatus();
      if (announce) notify('Cannot reach the library', error.message);
      return false;
    }
  }

  /* ---------------------------------------------------------------- delete */
  function openDelete(pretty, filename) {
    if (busy) return;
    pendingDelete = filename;
    $('deleteTarget').textContent = pretty;
    $('deleteDialog').showModal();
    $('confirmDelete').focus();
  }

  async function confirmDelete() {
    if (busy || !pendingDelete) return;
    const filename = pendingDelete;
    const pretty = displayName(filename);
    $('deleteDialog').close();
    busy = 'delete';
    controls();
    clearNotice();
    try {
      const data = await api(`/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' }, 60000);
      history = [];
      answerExists = false;
      $('answer').hidden = true;
      $('welcome').hidden = false;
      $('answerText').textContent = '';
      $('sources').replaceChildren();
      $('passages').replaceChildren();
      await refresh();
      notify(
        'Document deleted',
        `${pretty} was removed (${data.chunks_removed} chunk${data.chunks_removed === 1 ? '' : 's'}).`,
        'success'
      );
    } catch (error) {
      notify('Could not delete the document', error.message);
    } finally {
      pendingDelete = null;
      busy = '';
      controls();
    }
  }

  /* ------------------------------------------------------------------- ask */
  // Small, safe Markdown renderer: builds DOM nodes only, never HTML strings.
  function renderInline(parent, text, depth = 0) {
    if (depth > 8) {
      parent.append(document.createTextNode(text));
      return;
    }
    const tokens = /(`+)([^`\n]+?)\1|\*\*([^\n]+?)\*\*|__([^\n]+?)__|\*([^*\n]+?)\*|_([^_\n]+?)_/g;
    let match;
    let last = 0;
    while ((match = tokens.exec(text))) {
      if (
        (match[4] || match[6]) &&
        (/[\w]/.test(text[match.index - 1] || '') || /[\w]/.test(text[tokens.lastIndex] || ''))
      ) {
        continue; // underscores inside identifiers/filenames are literal
      }
      parent.append(document.createTextNode(text.slice(last, match.index)));
      const tag = match[2] ? 'code' : match[3] || match[4] ? 'strong' : 'em';
      const el = node(tag, '');
      const content = match[2] || match[3] || match[4] || match[5] || match[6];
      if (tag === 'code') el.textContent = content;
      else renderInline(el, content, depth + 1);
      parent.append(el);
      last = tokens.lastIndex;
    }
    parent.append(document.createTextNode(text.slice(last)));
  }

  function renderAnswer(text) {
    const root = $('answerText');
    root.replaceChildren();
    const lines = String(text).replace(/\r\n?/g, '\n').split('\n');
    let i = 0;
    const isBlock = (line) =>
      /^\s*(?:#{1,6}\s|[-+*]\s|\d+[.)]\s|>\s?|```|~~~|(?:-{3,}|\*{3,}|_{3,})\s*$)/.test(line);

    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) {
        i++;
        continue;
      }
      const fence = line.match(/^\s*(`{3,}|~{3,})/);
      if (fence) {
        const code = node('code', '');
        const pre = node('pre', '');
        const content = [];
        i++;
        while (i < lines.length && !lines[i].trim().startsWith(fence[1])) content.push(lines[i++]);
        if (i < lines.length) i++;
        code.textContent = content.join('\n');
        pre.append(code);
        root.append(pre);
        continue;
      }
      const heading = line.match(/^\s*(#{1,6})\s+(.+?)\s*#*$/);
      if (heading) {
        const el = node('h' + Math.min(6, heading[1].length + 2), '');
        renderInline(el, heading[2]);
        root.append(el);
        i++;
        continue;
      }
      if (/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
        root.append(node('hr', ''));
        i++;
        continue;
      }
      if (/^\s*>/.test(line)) {
        const el = node('blockquote', '');
        const content = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) content.push(lines[i++].replace(/^\s*>\s?/, ''));
        renderInline(el, content.join('\n'));
        root.append(el);
        continue;
      }
      const listMatch = line.match(/^\s*(?:([-+*])|(\d+)[.)])\s+(.+)$/);
      if (listMatch) {
        const ordered = !!listMatch[2];
        const list = node(ordered ? 'ol' : 'ul', '');
        if (ordered) list.start = Number(listMatch[2]);
        while (i < lines.length) {
          const item = lines[i].match(/^\s*(?:([-+*])|(\d+)[.)])\s+(.+)$/);
          if (!item || !!item[2] !== ordered) break;
          const li = node('li', '');
          let content = item[3];
          i++;
          while (i < lines.length && /^\s+\S/.test(lines[i]) && !isBlock(lines[i])) content += '\n' + lines[i++].trim();
          renderInline(li, content);
          list.append(li);
        }
        root.append(list);
        continue;
      }
      const paragraph = node('p', '');
      const content = [line];
      i++;
      while (i < lines.length && lines[i].trim() && !isBlock(lines[i])) content.push(lines[i++]);
      renderInline(paragraph, content.join('\n'));
      root.append(paragraph);
    }
  }

  function renderSources(sources) {
    $('sources').replaceChildren();
    for (const source of sources) {
      const citation = source.citation || source.doc_name || 'Source';
      const label = source.doc_name
        ? `${displayName(source.doc_name)} · ${source.page_number ? 'Page ' + source.page_number : 'Full text'}`
        : citation;
      const el = node('span', 'source', label);
      el.title = citation;
      $('sources').append(el);
    }
    if (!sources.length) $('sources').append(node('span', 'source', 'No sources returned'));
  }

  async function ask(event) {
    event.preventDefault();
    if ($('askButton').disabled || busy) return;
    const question = $('questionInput').value.trim();
    busy = 'ask';
    clearNotice();
    switchView('ask');
    $('welcome').hidden = true;
    $('loading').hidden = false;
    $('answer').hidden = true;
    controls();

    try {
      const payload = { question };
      if (history.length) payload.history = history.slice(-2 * HISTORY_TURNS);
      const data = await post('/ask', payload, 90000);
      if (typeof data.answer !== 'string') throw new Error('The server did not return an answer.');

      $('askedQuestion').textContent = question;
      renderAnswer(data.answer);
      renderSources(data.sources || []);

      const seen = new Set();
      const chunks = (data.retrieved_context || []).filter((chunk) => {
        if (!chunk.text || seen.has(chunk.text)) return false;
        seen.add(chunk.text);
        return true;
      });
      $('contextSummary').textContent = `Explore ${chunks.length} retrieved passage${chunks.length === 1 ? '' : 's'}`;
      $('contextSummary').parentElement.open = false;
      $('passages').replaceChildren();
      for (const chunk of chunks) {
        const card = node('details', 'passage');
        card.append(
          node(
            'summary',
            '',
            `${displayName(chunk.doc_name)} · ${chunk.page_number ? 'Page ' + chunk.page_number : 'Full text'} · Cosine similarity ${Number(
              chunk.similarity_score
            ).toFixed(4)}`
          ),
          node('p', '', chunk.text)
        );
        $('passages').append(card);
      }

      history.push({ role: 'user', content: question }, { role: 'assistant', content: data.answer });
      history = history.slice(-2 * HISTORY_TURNS);
      answerExists = true;
      $('answer').hidden = false;
    } catch (error) {
      notify('Could not generate an answer', error.message + (answerExists ? '\nYour previous answer is kept below.' : ''));
      $('answer').hidden = !answerExists;
      $('welcome').hidden = answerExists;
    } finally {
      busy = '';
      $('loading').hidden = true;
      controls();
    }
  }

  /* --------------------------------------------------------------- docking */
  function resizeDock() {
    requestAnimationFrame(() =>
      document.documentElement.style.setProperty('--dock-height', $('bottomDock').offsetHeight + 40 + 'px')
    );
  }

  function growInput() {
    const input = $('questionInput');
    input.style.height = 'auto';
    input.style.height = Math.min(120, Math.max(40, input.scrollHeight)) + 'px';
    resizeDock();
  }

  /* ---------------------------------------------------------------- events */
  $('dropZone').onclick = () => {
    if (!busy) $('fileInput').click();
  };
  $('heroUpload').onclick = () => {
    if (!busy) $('fileInput').click();
  };
  $('fileInput').onchange = (event) => select(event.target.files);

  let dragDepth = 0;
  $('dropZone').ondragenter = (event) => {
    event.preventDefault();
    if (!busy) {
      dragDepth++;
      $('dropZone').classList.add('drag-active');
    }
  };
  $('dropZone').ondragover = (event) => event.preventDefault();
  $('dropZone').ondragleave = (event) => {
    event.preventDefault();
    if (--dragDepth <= 0) $('dropZone').classList.remove('drag-active');
  };
  $('dropZone').ondrop = (event) => {
    event.preventDefault();
    dragDepth = 0;
    $('dropZone').classList.remove('drag-active');
    select(event.dataTransfer.files);
  };
  for (const type of ['dragover', 'drop']) {
    window.addEventListener(type, (event) => {
      if ([...event.dataTransfer.types].includes('Files')) event.preventDefault();
    });
  }

  $('indexButton').onclick = indexFiles;
  $('questionForm').onsubmit = ask;
  $('refreshButton').onclick = async () => {
    if (busy) return;
    busy = 'refresh';
    controls();
    clearNotice();
    const ok = await refresh(true);
    if (ok) notify('Library refreshed', `${status.documents_count} document(s) indexed.`, 'success');
    busy = '';
    controls();
  };
  $('dismissNotice').onclick = clearNotice;
  $('cancelDelete').onclick = () => {
    pendingDelete = null;
    $('deleteDialog').close();
  };
  $('confirmDelete').onclick = confirmDelete;
  $('deleteDialog').addEventListener('cancel', () => {
    pendingDelete = null;
  });

  document.querySelectorAll('[data-view]').forEach((button) => (button.onclick = () => switchView(button.dataset.view)));
  $('manageLibrary').onclick = () => switchView('library');
  document.querySelectorAll('[data-question]').forEach(
    (button) =>
      (button.onclick = () => {
        if (busy) return;
        $('questionInput').value = button.dataset.question;
        growInput();
        controls();
        $('questionInput').focus();
      })
  );

  $('questionInput').oninput = () => {
    growInput();
    controls();
  };
  $('questionInput').onkeydown = (event) => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey) && !event.isComposing) {
      event.preventDefault();
      if (!$('askButton').disabled) $('questionForm').requestSubmit();
    }
  };
  $('copyButton').onclick = async () => {
    try {
      await navigator.clipboard.writeText($('answerText').innerText);
      clearTimeout(copyTimer);
      $('copyButton').textContent = 'Copied ✓';
      copyTimer = setTimeout(() => ($('copyButton').textContent = 'Copy answer'), 1800);
    } catch {
      notify('Copy is unavailable', 'Your browser blocked clipboard access. Select the answer text and copy it manually.');
    }
  };

  window.addEventListener('beforeunload', (event) => {
    if (busy === 'index') {
      event.preventDefault();
      event.returnValue = '';
    }
  });

  if (window.visualViewport) {
    const adjust = () => {
      const vv = window.visualViewport;
      const focused = document.activeElement === $('questionInput');
      const inset = Math.max(0, window.innerHeight - vv.height - vv.offsetTop);
      document.body.classList.toggle('keyboard-open', focused && inset > 120 && vv.scale < 1.1);
      document.documentElement.style.setProperty('--keyboard-inset', inset + 'px');
      resizeDock();
    };
    visualViewport.addEventListener('resize', adjust);
    visualViewport.addEventListener('scroll', adjust);
    $('questionInput').addEventListener('focus', adjust);
    $('questionInput').addEventListener('blur', () => setTimeout(adjust, 100));
  }

  new ResizeObserver(resizeDock).observe($('bottomDock'));

  refresh();
  growInput();
});
