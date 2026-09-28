/* CYBER-RAG frontend. Same Flask API, no credentials or external dependencies. */
'use strict';
document.addEventListener('DOMContentLoaded', () => {
 const $ = id => document.getElementById(id);
 const GUARD_KEY='cyber-rag-index-operation-v2';
 let queue=[], status=null, busy='', view='library', answerExists=false, copyTimer;
 let uncertain=readGuard(), persistent=true;
 const titles={library:['Library','Everything you know. Ready to explore.'],ask:['Ask','A little curiosity goes a long way.'],activity:['Activity','Behind every answer, a little intelligence.']};
 function node(tag,cls,text){const el=document.createElement(tag);if(cls)el.className=cls;if(text!=null)el.textContent=text;return el;}
 function icon(name,cls=''){const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('class',cls);svg.setAttribute('aria-hidden','true');const use=document.createElementNS(svg.namespaceURI,'use');use.setAttribute('href','#'+name);svg.append(use);return svg;}
 function readGuard(){try{return localStorage.getItem(GUARD_KEY);}catch{return null;}}
 function guard(value){uncertain=value;try{value?localStorage.setItem(GUARD_KEY,value):localStorage.removeItem(GUARD_KEY);}catch{persistent=false;}controls();}
 function log(text){const lines=$('activity').textContent.split('\n');lines.push(`${new Date().toLocaleTimeString()} · ${text}`);$('activity').textContent=lines.slice(-120).join('\n');$('activity').scrollTop=$('activity').scrollHeight;}
 function notify(title,text='',kind='error',reveal=true){$('noticeTitle').textContent=title;$('noticeText').textContent=text;$('notice').className='notice '+kind;$('notice').hidden=false;}
 function clearNotice(){$('notice').hidden=true;}
 function recoveryNotice(){notify('Index state needs attention','An indexing or reset request did not finish cleanly. Asking and indexing are paused to avoid duplicates or mismatched sources. In Activity, verify the server has finished, then clear the index and re-upload your documents.','error',false);}
 function switchView(next,scroll=true){if(!titles[next])return;view=next;for(const key of Object.keys(titles))$(key+'View').hidden=key!==next;$('viewTitle').textContent=titles[next][0];$('viewSubtitle').textContent=titles[next][1];document.title=`${titles[next][0]} · CYBER-RAG`;document.querySelectorAll('[data-view]').forEach(b=>{const active=b.dataset.view===next;b.classList.toggle('selected',active);active?b.setAttribute('aria-current','page'):b.removeAttribute('aria-current');});if(scroll)window.scrollTo({top:0,behavior:'instant'});}
 function controls(){
  const locked=!!busy;
  for(const id of ['dropZone','fileInput','heroUpload','resetButton','refreshButton'])$(id).disabled=locked;
  $('indexButton').disabled=locked||!!uncertain||!status||!queue.length;
  $('askButton').disabled=locked||!!uncertain||!status?.total_chunks_indexed||!status?.is_groq_key_set||!$('questionInput').value.trim();
  $('questionInput').readOnly=busy==='ask';
  document.querySelectorAll('.queue-row button,[data-question]').forEach(b=>b.disabled=locked);
  $('composerHint').textContent=busy==='ask'?'Generating an answer…':busy==='index'?'Processing documents. Keep this page open.':busy==='reset'?'Clearing the index…':uncertain?'Index needs recovery · open Activity':!status?'Server unavailable · refresh in Library':!status.total_chunks_indexed?'Add and index documents in Library':!status.is_groq_key_set?'Set GROQ_API_KEY in your .env, then restart Flask':`Ask across ${status.documents_count} document${status.documents_count===1?'':'s'}`;
  $('askContextText').textContent=!status?'Server unavailable':uncertain?'Index needs recovery':`${status.documents_count} document${status.documents_count===1?'':'s'} in your library`;
  $('questionForm').setAttribute('aria-busy',String(busy==='ask'));
  $('libraryView').setAttribute('aria-busy',String(busy==='index'));
  resizeDock();
 }
 async function api(path,options={},timeout=15000){
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),timeout);
  try{
   const res=await fetch(path,{...options,signal:controller.signal,cache:'no-store'});
   let data;try{data=await res.json();}catch{throw new Error(res.status===413?'The entire upload request exceeds 16 MB. Remove a file or use a smaller batch.':`Unexpected server response (${res.status}). Check the Flask console.`);}
   if(!res.ok){const extra=[...(data.details||[]),...(data.report||[]).filter(r=>r.status==='failed').map(r=>`${r.filename}: ${r.reason}`)];throw new Error([data.error||`Request failed (${res.status}).`,...extra].join('\n'));}
   return data;
  }catch(e){if(e.name==='AbortError')throw new Error('The server took too long to respond. It may still be processing; this is not a cancellation.');if(e instanceof TypeError)throw new Error('Could not reach Flask. Check your connection and that the server is running.');throw e;}finally{clearTimeout(timer);}
 }
 const post=(path,data,timeout)=>api(path,{method:'POST',...(data?{headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}:{})},timeout);
 function renderStatus(){
  $('connection').classList.toggle('online',!!status);
  $('statusText').textContent=!status?'Offline':uncertain?'Check index':status.is_groq_key_set?'Connected':'Key needed';
  $('docsCount').textContent=status?status.documents_count:'—';$('chunksCount').textContent=status?status.total_chunks_indexed:'—';
  const list=$('documentList');list.replaceChildren();
  if(!status||(status.indexed_documents||[]).length===0){const empty=node('div','library-empty');empty.append(icon('doc'),node('strong','',status?'Your library starts here':'Library unavailable'),node('span','',status?'Add a PDF or TXT file, then index it.\nYour documents will appear here.':'Check that Flask is running, then select Refresh.'));list.append(empty);}
  for(const doc of status?.indexed_documents||[]){const row=node('div','document'),info=node('div','document-info'),txt=doc.filename.toLowerCase().endsWith('.txt');row.append(node('span','file-icon'+(txt?' txt':''),txt?'TXT':'PDF'));info.append(node('strong','',displayName(doc.filename)),node('small','',`${doc.pages} text section${doc.pages===1?'':'s'} · ${doc.chunks} chunk${doc.chunks===1?'':'s'}`));info.title='Stored as: '+doc.filename;row.append(info,icon('check','indexed-mark'));list.append(row);}
  controls();
 }
 async function refresh(announce=false){try{const data=await api('/status');if(!Array.isArray(data.indexed_documents)||!Number.isFinite(data.total_chunks_indexed))throw new Error('Invalid status response. Check the Flask /status endpoint.');status=data;renderStatus();if(announce)log('Workspace status refreshed.');return true;}catch(e){status=null;renderStatus();log(e.message);if(announce)notify('Cannot refresh the library',e.message);return false;}}
 function safeBase(name){let s=name.normalize('NFKD').replace(/[^\x00-\x7F]/g,'').replace(/[\\/]/g,' ').split(/\s+/).join('_').replace(/[^A-Za-z0-9_.-]/g,'').replace(/^[._]+|[._]+$/g,'');const dot=s.lastIndexOf('.');let stem=(dot>0?s.slice(0,dot):'document').slice(0,90)||'document';if(/^(con|prn|aux|nul|com[0-9]|lpt[0-9])$/i.test(stem))stem='_'+stem;const ext=name.split('.').pop().toLowerCase();return stem+'.'+ext;}
 function displayName(name){return name.replace(/__rag_[a-f0-9]{16}(?=\.(pdf|txt)$)/i,'');}
 function key(name){return safeBase(displayName(name)).toLowerCase();}
 function storedName(name){const bytes=new Uint8Array(8);crypto.getRandomValues(bytes);const token=Array.from(bytes,b=>b.toString(16).padStart(2,'0')).join('');const dot=name.lastIndexOf('.');return name.slice(0,dot)+'__rag_'+token+name.slice(dot);}
 const bytes=n=>n>=1048576?(n/1048576).toFixed(1)+' MB':Math.max(1,Math.round(n/1024))+' KB';
 function renderQueue(){
  $('queue').replaceChildren();if(!queue.length&&parseFloat($('queue').style.minHeight)>0)$('queue').append(node('li','queue-complete','✓  Files added to your library'));$('queueHeading').hidden=!queue.length&&!parseFloat($('queue').style.minHeight);$('queueHeading').querySelector('h4').textContent=queue.length?'Ready to upload':'Upload complete';$('queueSize').textContent=bytes(queue.reduce((n,item)=>n+item.file.size,0));
  queue.forEach((item,i)=>{const row=node('li','queue-row'),info=node('div');info.append(node('strong','',item.file.name),node('small','',item.failure||`${bytes(item.file.size)} · ${item.base}`));const b=node('button','','×');b.type='button';b.setAttribute('aria-label',`Remove ${item.file.name}`);b.onclick=()=>{if(busy)return;queue.splice(i,1);$('queue').style.minHeight='';renderQueue();};row.append(info,b);$('queue').append(row);});controls();
 }
 function select(files){
  if(busy)return;$('queue').style.minHeight='';const errors=[];
  for(const file of files){
   if(!/\.(pdf|txt)$/i.test(file.name)){errors.push(`${file.name}: only PDF and TXT files are supported.`);continue;}
   if(!file.size){errors.push(`${file.name}: the file is empty.`);continue;}
   const base=safeBase(file.name),id=key(base);
   if(queue.some(f=>key(f.base)===id)){errors.push(`${file.name}: a file with the same normalized name is already queued.`);continue;}
   if(status?.indexed_documents.some(d=>key(d.filename)===id)){errors.push(`${file.name}: already in your library. Use a distinct filename for different content.`);continue;}
   if(queue.reduce((n,f)=>n+f.file.size,0)+file.size>16*1024*1024-65536){errors.push(`${file.name}: the batch would exceed 16 MB. Upload a smaller batch.`);continue;}
   queue.push({file,base,stored:storedName(base),uploaded:false,failure:''});
  }
  $('fileInput').value='';renderQueue();if(errors.length)notify('Some files were not added',errors.join('\n'));else clearNotice();
 }
 // Web Locks coordinate these mutations across tabs in this origin, when supported.
 async function exclusive(task){if(navigator.locks)return navigator.locks.request('cyber-rag-mutation',{ifAvailable:true},async lock=>{if(!lock){notify('Another tab is updating the index','Wait for it to finish, then refresh this library.');return;}return task();});return task();}
 async function indexFiles(){
  if(busy||!queue.length||uncertain||!status)return;
  await exclusive(async()=>{
   const existing=readGuard();if(existing){uncertain=existing;recoveryNotice();controls();return;}
   busy='index';controls();clearNotice();$('indexProgress').classList.add('processing');$('queue').style.minHeight=$('queue').offsetHeight+'px';
   let started=false;const reports=[];
   try{
    // Check names again in case another tab indexed a file since selection.
    const fresh=await api('/status');status=fresh;
    if(queue.some(q=>fresh.indexed_documents.some(d=>key(d.filename)===key(q.base))))throw new Error('A queued filename is now indexed. Remove that file from the queue before continuing.');
    const pending=queue.filter(q=>!q.uploaded);
    if(pending.length){$('indexButton').firstElementChild.textContent='Uploading…';$('indexProgressText').textContent='Uploading your documents…';const form=new FormData();pending.forEach(q=>form.append('files',q.file,q.stored));log(`Uploading ${pending.length} file(s).`);const data=await api('/upload',{method:'POST',body:form},120000);if(!Array.isArray(data.saved_files))throw new Error('Upload response is missing saved filenames.');pending.forEach(q=>{q.uploaded=data.saved_files.includes(q.stored);if(!q.uploaded)q.failure='Not accepted by the server.';});reports.push(...(data.warnings||[]));}
    const accepted=queue.filter(q=>q.uploaded);if(!accepted.length)throw new Error('No files were accepted by the server.');
    guard('index:'+new Date().toISOString());started=true;
    $('indexButton').firstElementChild.textContent='Indexing…';$('indexProgressText').textContent='Extracting, embedding & indexing. Waiting for the server…';log('Indexing started. No intermediate progress is available from Flask.');
    const data=await post('/index',{filenames:accepted.map(q=>q.stored)},180000);
    if(!Array.isArray(data.report)||!Number.isFinite(data.new_chunks_added)||!accepted.every(q=>data.report.some(r=>r.filename===q.stored&&['success','failed'].includes(r.status))))throw new Error('The indexing response was incomplete. Its outcome cannot be confirmed.');
    // In the supplied backend, per-file failures happen before that file is added to the index.
    const successes=new Set(data.report.filter(r=>r.status==='success').map(r=>r.filename));
    queue=queue.filter(q=>!successes.has(q.stored));
    for(const q of queue){const fail=data.report.find(r=>r.filename===q.stored&&r.status==='failed');if(fail){q.failure=fail.reason||'Could not read the document.';reports.push(`${q.file.name}: ${q.failure}`);}}
    data.report.forEach(r=>log(`${displayName(r.filename)}: ${r.status}${r.reason?' — '+r.reason:''}`));
    $('indexProgressText').textContent=reports.length?'Some files need attention.':'Indexed successfully. Your library is ready.';guard(null);notify(reports.length?'Some documents need attention':'Documents indexed',reports.length?reports.join('\n'):`Added ${data.new_chunks_added} chunks. Open Ask to explore your library.`,reports.length?'error':'success');log(`Added ${data.new_chunks_added} chunks.`);
   }catch(e){$('indexProgressText').textContent='Processing stopped. Check the message above.';log(e.message);if(started){notify('Indexing outcome needs recovery',e.message+'\nDo not retry this batch. Verify the server has finished, then clear the index in Activity.');}else notify('Could not upload documents',e.message);}
   finally{await refresh();busy='';$('indexProgress').classList.remove('processing');$('indexButton').firstElementChild.textContent='Index documents';renderQueue();if(!persistent&&uncertain)log('Storage is unavailable: recovery state cannot survive closing this page.');}
  });
 }
 // A deliberately small, safe Markdown renderer for model answers.
 // Creates DOM nodes only: model HTML, URLs and scripts are never executed.
 function answerInline(parent, text, depth=0){
  if(depth>8){parent.append(document.createTextNode(text));return;}
  const tokens=/(`+)([^`\n]+?)\1|\*\*([^\n]+?)\*\*|__([^\n]+?)__|\*([^*\n]+?)\*|_([^_\n]+?)_/g;
  let match,last=0;
  while((match=tokens.exec(text))){
   // Underscores embedded in identifiers/filenames are literal, not emphasis.
   if((match[4]||match[6])&&(/[\w]/.test(text[match.index-1]||'')||/[\w]/.test(text[tokens.lastIndex]||'')))continue;
   parent.append(document.createTextNode(text.slice(last,match.index)));
   const tag=match[2]?'code':(match[3]||match[4])?'strong':'em';
   const el=node(tag,'');const content=match[2]||match[3]||match[4]||match[5]||match[6];
   if(tag==='code')el.textContent=content;else answerInline(el,content,depth+1);
   parent.append(el);last=tokens.lastIndex;
  }
  parent.append(document.createTextNode(text.slice(last)));
 }
 function renderAnswer(text){
  const root=$('answerText');root.replaceChildren();
  const lines=String(text).replace(/\r\n?/g,'\n').split('\n');let i=0;
  const isBlock=line=>/^\s*(?:#{1,6}\s|[-+*]\s|\d+[.)]\s|>\s?|```|~~~|(?:-{3,}|\*{3,}|_{3,})\s*$)/.test(line);
  while(i<lines.length){
   const line=lines[i];if(!line.trim()){i++;continue;}
   const fence=line.match(/^\s*(`{3,}|~{3,})/);
   if(fence){const code=node('code',''),pre=node('pre','');const content=[];i++;while(i<lines.length&&!lines[i].trim().startsWith(fence[1]))content.push(lines[i++]);if(i<lines.length)i++;code.textContent=content.join('\n');pre.append(code);root.append(pre);continue;}
   const heading=line.match(/^\s*(#{1,6})\s+(.+?)\s*#*$/);
   if(heading){const el=node('h'+Math.min(6,heading[1].length+2),'');answerInline(el,heading[2]);root.append(el);i++;continue;}
   if(/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)){root.append(node('hr',''));i++;continue;}
   if(/^\s*>/.test(line)){const el=node('blockquote','');const content=[];while(i<lines.length&&/^\s*>/.test(lines[i]))content.push(lines[i++].replace(/^\s*>\s?/,''));answerInline(el,content.join('\n'));root.append(el);continue;}
   const listMatch=line.match(/^\s*(?:([-+*])|(\d+)[.)])\s+(.+)$/);
   if(listMatch){const ordered=!!listMatch[2],list=node(ordered?'ol':'ul','');if(ordered)list.start=Number(listMatch[2]);
    while(i<lines.length){const item=lines[i].match(/^\s*(?:([-+*])|(\d+)[.)])\s+(.+)$/);if(!item||!!item[2]!==ordered)break;const li=node('li','');let content=item[3];i++;while(i<lines.length&&/^\s+\S/.test(lines[i])&&!isBlock(lines[i]))content+='\n'+lines[i++].trim();answerInline(li,content);list.append(li);}
    root.append(list);continue;
   }
   const paragraph=node('p','');const content=[line];i++;while(i<lines.length&&lines[i].trim()&&!isBlock(lines[i]))content.push(lines[i++]);answerInline(paragraph,content.join('\n'));root.append(paragraph);
  }
 }
 function renderSources(sources){
  $('sources').replaceChildren();
  for(const source of sources){
   const citation=source.citation||source.doc_name||'Source';
   // Use structured fields rather than trying to interpret the LLM's citations.
   const label=source.doc_name?`${displayName(source.doc_name)} · ${source.page_number?'Page '+source.page_number:'Full text'}`:citation;
   const el=node('span','source',label);el.title=citation;$('sources').append(el);
  }
  if(!sources.length)$('sources').append(node('span','source','No sources returned'));
 }

 async function ask(event){
  event.preventDefault();if($('askButton').disabled||busy)return;
  const question=$('questionInput').value.trim();busy='ask';clearNotice();switchView('ask');$('welcome').hidden=true;$('loading').hidden=false;$('answer').hidden=true;controls();
  try{const data=await post('/ask',{question},90000);if(typeof data.answer!=='string')throw new Error('The server did not return an answer.');$('askedQuestion').textContent=question;renderAnswer(data.answer);renderSources(data.sources||[]);const seen=new Set(),chunks=(data.retrieved_context||[]).filter(c=>{if(seen.has(c.text))return false;seen.add(c.text);return true;});$('contextSummary').textContent=`Explore ${chunks.length} retrieved passage${chunks.length===1?'':'s'}`;$('contextSummary').parentElement.open=false;$('passages').replaceChildren();for(const c of chunks){const card=node('details','passage');card.append(node('summary','',`${displayName(c.doc_name)} · ${c.page_number?'Page '+c.page_number:'Full text'} · Cosine similarity ${Number(c.similarity_score).toFixed(4)}`),node('p','',c.text));$('passages').append(card);}answerExists=true;$('answer').hidden=false;log('Answer received with retrieved source context.');}
  catch(e){notify('Could not generate an answer',e.message+(answerExists?'\nYour previous answer is kept below.':''));$('answer').hidden=!answerExists;$('welcome').hidden=answerExists;log(e.message);}
  finally{busy='';$('loading').hidden=true;controls();}
 }
 function openReset(){if(busy)return;$('resetWarning').hidden=!uncertain;$('resetCheckLabel').hidden=!uncertain;$('resetCheck').checked=false;$('confirmReset').disabled=!!uncertain;$('resetDialog').showModal();}
 async function reset(){
  if(busy||(uncertain&&!$('resetCheck').checked))return;
  await exclusive(async()=>{
   $('resetDialog').close();busy='reset';controls();clearNotice();guard('reset:'+new Date().toISOString());
   try{await post('/reset',null,30000);guard(null);queue=[];$('queue').style.minHeight='';$('indexProgressText').textContent='Choose files to add to your library.';status=null;answerExists=false;$('answer').hidden=true;$('welcome').hidden=false;$('askedQuestion').textContent='';$('answerText').textContent='';$('sources').replaceChildren();$('passages').replaceChildren();$('questionInput').value='';growInput();log('Index and upload staging cleared.');await refresh();notify('Workspace cleared','Add documents in Library to start again.','success');}
   catch(e){notify('Reset could not be confirmed',e.message+'\nRecovery mode remains active. Check the Flask console before trying again.');log(e.message);}
   finally{busy='';renderQueue();renderStatus();}
  });
 }
 async function refreshAction(){if(busy)return;busy='refresh';controls();clearNotice();await refresh(true);busy='';if(uncertain)recoveryNotice();controls();}
 function resizeDock(){requestAnimationFrame(()=>document.documentElement.style.setProperty('--dock-height',($('bottomDock').offsetHeight+40)+'px'));}
 function growInput(){$('questionInput').style.height='auto';$('questionInput').style.height=Math.min(120,Math.max(40,$('questionInput').scrollHeight))+'px';resizeDock();}
 $('dropZone').onclick=()=>{if(!busy)$('fileInput').click();};$('heroUpload').onclick=()=>{if(!busy)$('fileInput').click();};$('fileInput').onchange=e=>select(e.target.files);
 let dragDepth=0;
 $('dropZone').ondragenter=e=>{e.preventDefault();if(!busy){dragDepth++;$('dropZone').classList.add('drag-active');}};
 $('dropZone').ondragover=e=>e.preventDefault();$('dropZone').ondragleave=e=>{e.preventDefault();if(--dragDepth<=0)$('dropZone').classList.remove('drag-active');};$('dropZone').ondrop=e=>{e.preventDefault();dragDepth=0;$('dropZone').classList.remove('drag-active');select(e.dataTransfer.files);};
 for(const type of ['dragover','drop'])window.addEventListener(type,e=>{if([...e.dataTransfer.types].includes('Files'))e.preventDefault();});
 $('indexButton').onclick=indexFiles;$('questionForm').onsubmit=ask;$('refreshButton').onclick=refreshAction;$('dismissNotice').onclick=clearNotice;$('resetButton').onclick=openReset;$('cancelReset').onclick=()=>$('resetDialog').close();$('resetCheck').onchange=()=>$('confirmReset').disabled=!!uncertain&&!$('resetCheck').checked;$('confirmReset').onclick=reset;
 document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>switchView(b.dataset.view));$('manageLibrary').onclick=()=>switchView('library');
 document.querySelectorAll('[data-question]').forEach(b=>b.onclick=()=>{if(busy)return;$('questionInput').value=b.dataset.question;growInput();controls();$('questionInput').focus();});
 $('questionInput').oninput=()=>{growInput();controls();};$('questionInput').onkeydown=e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)&&!e.isComposing){e.preventDefault();if(!$('askButton').disabled)$('questionForm').requestSubmit();}};
 $('copyButton').onclick=async()=>{try{await navigator.clipboard.writeText($('answerText').innerText);clearTimeout(copyTimer);$('copyButton').textContent='Copied ✓';copyTimer=setTimeout(()=>$('copyButton').textContent='Copy answer',1800);}catch{notify('Copy is unavailable','Your browser has blocked clipboard access. Select the answer text and copy it manually.');}};
 window.addEventListener('storage',e=>{if(e.key===GUARD_KEY){uncertain=e.newValue;if(uncertain)recoveryNotice();else if(!busy)refresh();controls();}});
 window.addEventListener('beforeunload',e=>{if(busy==='index'||busy==='reset'){e.preventDefault();e.returnValue='';}});
 if(window.visualViewport){const adjust=()=>{const vv=window.visualViewport;const focused=document.activeElement===$('questionInput');const inset=Math.max(0,window.innerHeight-vv.height-vv.offsetTop);const keyboard=focused&&inset>120&&vv.scale<1.1;document.body.classList.toggle('keyboard-open',keyboard);document.documentElement.style.setProperty('--keyboard-inset',inset+'px');resizeDock();};visualViewport.addEventListener('resize',adjust);visualViewport.addEventListener('scroll',adjust);$('questionInput').addEventListener('focus',adjust);$('questionInput').addEventListener('blur',()=>setTimeout(adjust,100));}
 new ResizeObserver(resizeDock).observe($('bottomDock'));
 if(uncertain)recoveryNotice();refresh();growInput();
});
