const $=(sel)=>document.querySelector(sel);
const loginView=$("#login-view");
const chatView=$("#chat-view");
const messagesEl=$("#messages");
const questionInput=$("#question-input");
const sendBtn=$("#send-btn");
const threadListEl=$("#thread-list");
const TOKEN_KEY="paper_agent_token";
const USERNAME_KEY="paper_agent_username";
const THREAD_KEY="paper_agent_thread";
let token=sessionStorage.getItem(TOKEN_KEY)||"";
//当前会话存本地，刷新后能回到同一个会话并从服务端拉回历史
let activeThread=localStorage.getItem(THREAD_KEY)||"";
let streaming=false;
//暴露一个可观测标志，便于外部脚本判断"这一轮是否还在生成"
Object.defineProperty(window,"__streaming",{get:()=>streaming});

function showLogin(){
  loginView.hidden=false;
  chatView.hidden=true;
}

function showChat(){
  loginView.hidden=true;
  chatView.hidden=false;
  $("#user-label").textContent=sessionStorage.getItem(USERNAME_KEY)||"";
  //恢复上次的收起状态
  chatView.classList.toggle("sidebar-collapsed",localStorage.getItem("paper_agent_sidebar")==="1");
  questionInput.focus();
  if(window.lucide){lucide.createIcons();}
  refreshThreads();
  loadHistory();
}

function logout(){
  sessionStorage.removeItem(TOKEN_KEY);
  sessionStorage.removeItem(USERNAME_KEY);
  localStorage.removeItem(THREAD_KEY);
  token="";
  activeThread="";
  showLogin();
}

function authHeaders(extra={}){
  return Object.assign({"Authorization":"Bearer "+token},extra);
}

let threadCache=[];
//图标统一用 lucide 的 data-lucide 占位，插入后由 createIcons 替换成 svg
function iconBtn(cls,icon,title,onClick){
  const b=document.createElement("button");
  b.type="button";
  b.className="msg-action "+cls;
  b.title=title;
  b.setAttribute("aria-label",title);
  b.innerHTML="<i data-lucide='"+icon+"'></i>";
  b.addEventListener("click",(e)=>{e.stopPropagation();onClick();});
  return b;
}

//操作按钮挂在消息右下角，鼠标移到该条消息上才显示
function attachActions(node,actions){
  const bar=document.createElement("div");
  bar.className="msg-actions";
  actions.forEach((a)=>bar.append(iconBtn(a.cls,a.icon,a.title,a.onClick)));
  node.el.append(bar);
  if(window.lucide){lucide.createIcons();}
}

//复制纯文本：优先用剪贴板API，非安全上下文回退到临时textarea
async function copyText(text){
  try{
    if(navigator.clipboard&&window.isSecureContext){
      await navigator.clipboard.writeText(text);
      return true;
    }
  }catch(ex){
    //落回下面的兜底方案
  }
  try{
    const ta=document.createElement("textarea");
    ta.value=text;
    ta.style.position="fixed";
    ta.style.opacity="0";
    document.body.append(ta);
    ta.select();
    const ok=document.execCommand("copy");
    ta.remove();
    return ok;
  }catch(ex){
    return false;
  }
}

//复制后把图标临时换成对勾，给个"已复制"的反馈
async function copyWithFeedback(btn,text){
  const ok=await copyText(text);
  const original=btn.innerHTML;
  btn.innerHTML="<i data-lucide='"+(ok?"check":"x")+"'></i>";
  if(window.lucide){lucide.createIcons();}
  setTimeout(()=>{
    btn.innerHTML=original;
    if(window.lucide){lucide.createIcons();}
  },1200);
}

//修改提问：原地变成输入框；确认后把改好的问题作为新提问追加到当前对话
//不就地改写原来的问答：那样要同时改动两条已在页面上的消息，容易和流式写入互相覆盖
function startEditUserMessage(node,originalText){
  if(streaming){return;}
  if(node.el.querySelector(".edit-box")){return;}
  const content=node.content;
  const saved=content.innerHTML;
  content.innerHTML="";
  const box=document.createElement("div");
  box.className="edit-box";
  const ta=document.createElement("textarea");
  ta.rows=2;
  ta.value=originalText;
  const actions=document.createElement("div");
  actions.className="edit-actions";
  const ok=document.createElement("button");
  ok.type="button";
  ok.className="btn btn-primary edit-ok";
  ok.innerHTML="<i data-lucide='send'></i><span>发送</span>";
  const cancel=document.createElement("button");
  cancel.type="button";
  cancel.className="btn btn-ghost edit-cancel";
  cancel.innerHTML="<i data-lucide='x'></i><span>取消</span>";
  actions.append(ok,cancel);
  box.append(ta,actions);
  content.append(box);
  ta.focus();
  ta.setSelectionRange(ta.value.length,ta.value.length);
  if(window.lucide){lucide.createIcons();}

  let closed=false;
  const close=(next)=>{
    if(closed){return;}
    closed=true;
    content.innerHTML=saved;
    if(next){
      questionInput.value=next;
      sendQuestion();
    }
  };
  ok.addEventListener("click",()=>close(ta.value.trim()));
  cancel.addEventListener("click",()=>close(""));
  ta.addEventListener("keydown",(e)=>{
    if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();close(ta.value.trim());}
    else if(e.key==="Escape"){e.preventDefault();close("");}
  });
}

//重跑一轮：模型输出前先把这条回答清空，重新走同一个流式链路
async function rerunMessage(node,question){
  if(streaming||!question){return;}
  node.el.classList.remove("has-error");
  const oldSources=node.el.querySelector(".sources");
  if(oldSources){oldSources.remove();}
  node.content.innerHTML="";
  node.answer="";
  node.question=question;
  setStatus(node.meta,"重新生成中",true);
  const ok=await streamAnswer(node,question);
  if(!ok){setStatus(node.meta,"连接中断",false);}
  streaming=false;
  sendBtn.disabled=!questionInput.value.trim();
  refreshThreads();
  attachAnswerActions(node,question);
}

//会话列表：标题由服务端取该会话第一条提问生成
async function refreshThreads(){
  try{
    const resp=await fetch("/threads",{headers:authHeaders()});
    if(resp.status===401){logout();return;}
    const data=await resp.json();
    threadCache=data.threads||[];
    renderThreadList(threadCache);
  }catch(ex){
    //列表拉取失败不影响问答主流程
  }
}

// 时间分组的锚点：日期按本地时区算，避免凌晨时段被分到前一天
const MS_DAY=86400000;

function parseUtc(s){
  if(!s){return null;}
  //后端存的是带时区的ISO串；没有时区标记时按UTC解析
  const t=Date.parse(/[Zz]|[+-]\d{2}:?\d{2}$/.test(s)?s:s+"Z");
  return Number.isNaN(t)?null:t;
}

function startOfDay(d){
  return new Date(d.getFullYear(),d.getMonth(),d.getDate()).getTime();
}

function groupLabel(updatedAt){
  const t=parseUtc(updatedAt);
  if(t===null){return "更早";}
  const today=startOfDay(new Date());
  const days=Math.floor((today-startOfDay(new Date(t)))/MS_DAY);
  if(days<=0){return "今天";}
  if(days===1){return "昨天";}
  if(days<7){return "近 7 天";}
  return "更早";
}

function renderThreadList(threads){
  threadListEl.innerHTML="";
  if(!threads.length){
    const empty=document.createElement("div");
    empty.className="thread-empty";
    empty.textContent="暂无历史对话";
    threadListEl.append(empty);
    return;
  }
  //按最后活动时间分组，组内保持服务端给的倒序
  const order=["今天","昨天","近 7 天","更早"];
  let lastGroup="";
  threads.forEach((t)=>{
    const group=groupLabel(t.updated_at);
    if(group!==lastGroup&&order.includes(group)){
      const head=document.createElement("div");
      head.className="thread-group";
      head.textContent=group;
      threadListEl.append(head);
      lastGroup=group;
    }
    const item=document.createElement("div");
    item.className="thread-item"+(t.thread_id===activeThread?" active":"");
    item.dataset.threadId=t.thread_id;

    const label=document.createElement("button");
    label.type="button";
    label.className="thread-label";
    label.textContent=t.title||"新对话";
    label.title=t.title||"";
    label.addEventListener("click",()=>selectThread(t.thread_id));
    //双击标题进入行内编辑，回车保存、Esc取消
    label.addEventListener("dblclick",(e)=>{
      e.preventDefault();
      startRename(item,label,t);
    });

    const del=document.createElement("button");
    del.type="button";
    del.className="thread-delete";
    del.title="删除对话";
    del.setAttribute("aria-label","删除对话");
    del.innerHTML="<i data-lucide='trash-2'></i>";
    del.addEventListener("click",(e)=>{
      e.stopPropagation();
      removeThread(t.thread_id);
    });

    item.append(label,del);
    threadListEl.append(item);
  });
  if(window.lucide){lucide.createIcons();}
}

function selectThread(threadId){
  if(streaming||!threadId||threadId===activeThread){return;}
  activeThread=threadId;
  localStorage.setItem(THREAD_KEY,threadId);
  renderThreadListFromCache();
  loadHistory();
}

function renderThreadListFromCache(){
  renderThreadList(threadCache);
}

function startNewThread(){
  if(streaming){return;}
  activeThread="";
  localStorage.removeItem(THREAD_KEY);
  messagesEl.innerHTML="";
  renderThreadListFromCache();
  questionInput.focus();
}

async function removeThread(threadId){
  if(streaming){return;}
  if(!window.confirm("删除这个对话？该操作不可恢复。")){return;}
  try{
    const resp=await fetch("/threads/"+encodeURIComponent(threadId),{method:"DELETE",headers:authHeaders()});
    if(resp.status===401){logout();return;}
  }catch(ex){
    return;
  }
  if(threadId===activeThread){
    activeThread="";
    localStorage.removeItem(THREAD_KEY);
    messagesEl.innerHTML="";
  }
  refreshThreads();
}

//行内改名：替换标题为输入框，回车提交、Esc取消、失焦按提交处理
function startRename(item,label,thread){
  if(item.querySelector(".thread-rename")){return;}
  const original=label.textContent;
  const input=document.createElement("input");
  input.type="text";
  input.className="thread-rename";
  input.value=original;
  input.maxLength=60;
  label.replaceWith(input);
  input.focus();
  input.select();

  let done=false;
  const finish=async(save)=>{
    if(done){return;}
    done=true;
    const next=input.value.trim();
    input.replaceWith(label);
    if(!save||next===original){
      return;
    }
    //改名失败就还原标题，不让界面和服务端不一致
    label.textContent=next||original;
    try{
      const resp=await fetch("/threads/"+encodeURIComponent(thread.thread_id),{
        method:"PATCH",
        headers:authHeaders({"Content-Type":"application/json"}),
        body:JSON.stringify({title:next})
      });
      if(resp.status===401){logout();return;}
      if(!resp.ok){
        label.textContent=original;
        return;
      }
      const data=await resp.json();
      label.textContent=data.title||next;
      label.title=label.textContent;
    }catch(ex){
      label.textContent=original;
    }
  };
  input.addEventListener("keydown",(e)=>{
    if(e.key==="Enter"){e.preventDefault();finish(true);}
    else if(e.key==="Escape"){e.preventDefault();finish(false);}
  });
  input.addEventListener("blur",()=>finish(true));
}

async function loadHistory(){
  messagesEl.innerHTML="";
  if(!activeThread){return;}
  try{
    const resp=await fetch("/threads/"+encodeURIComponent(activeThread)+"/messages",{headers:authHeaders()});
    if(resp.status===401){logout();return;}
    if(!resp.ok){return;}
    const data=await resp.json();
    renderHistory(data.messages||[]);
  }catch(ex){
    //历史加载失败就停在空状态，不影响继续提问
  }
}

//把库里存的对话还原到界面：user_query 是提问，assistant/task_result 是回答
function renderHistory(messages){
  //重新生成需要知道这条回答对应哪个提问，按出现顺序记住最近一条提问
  let lastUserQuestion="";
  messages.forEach((m)=>{
    if(m.message_type==="user_query"){
      lastUserQuestion=m.content||"";
      const node=addMessage("user",m.content||"");
      const text=m.content||"";
      attachActions(node,[
        {cls:"action-copy",icon:"copy",title:"复制",onClick:()=>copyWithFeedback(node.el.querySelector(".action-copy"),text)},
        {cls:"action-edit",icon:"pencil",title:"修改并重新提问",onClick:()=>startEditUserMessage(node,text)}
      ]);
      return;
    }
    if(m.message_type==="assistant_answer"||m.message_type==="task_result"){
      const node=addMessage("bot");
      if(m.sources&&m.sources.length){node.el.append(renderSources(m.sources));}
      const text=m.content||"";
      node.answer=text;
      node.content.innerHTML=renderMarkdown(text);
      const question=lastUserQuestion;
      attachActions(node,[
        {cls:"action-copy-answer",icon:"copy",title:"复制回答",
         onClick:()=>copyWithFeedback(node.el.querySelector(".action-copy-answer"),text)},
        {cls:"action-regen",icon:"refresh-cw",title:"重新生成",
         onClick:()=>rerunMessage(node,question)}
      ]);
      return;
    }
    if(m.message_type==="system_error"){
      const node=addMessage("bot");
      node.el.classList.add("has-error");
      setStatus(node.meta,m.content||"任务异常",false);
    }
  });
  messagesEl.scrollTop=messagesEl.scrollHeight;
}

function addMessage(role,text=""){
  const el=document.createElement("div");
  el.className=role==="user"?"msg msg-user":"msg msg-bot";
  const meta=document.createElement("div");
  meta.className="msg-meta";
  const content=document.createElement("div");
  content.className="msg-content";
  if(text){
    content.textContent=text;
  }
  el.append(meta,content);
  messagesEl.append(el);
  messagesEl.scrollTop=messagesEl.scrollHeight;
  return {el,meta,content};
}

function setRouteBadge(meta,route){
  meta.innerHTML="";
  const badge=document.createElement("span");
  badge.className=route==="survey"?"badge badge-survey":"badge badge-simple";
  badge.textContent=route==="survey"?"文献综述":"单篇问答";
  meta.append(badge);
}

function setStatus(meta,text,spin=false){
  const old=meta.querySelector(".status-line");
  if(old){old.remove();}
  if(!text){return;}
  const line=document.createElement("div");
  line.className="status-line";
  if(spin){
    const s=document.createElement("span");
    s.className="spinner";
    line.append(s);
  }
  const t=document.createElement("span");
  t.textContent=text;
  line.append(t);
  meta.append(line);
}

function escapeHtml(s){
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function escapeAttr(s){
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
          .replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}

//公式与普通文本分开处理：公式交给 KaTeX，文本再走转义，避免下划线被当成强调
//块级公式可能跨多行，所以要在按行渲染之前先把公式抽出来占位
//只匹配定界符本身，靠"成对出现"判断公式范围：
//用"开定界符+非贪婪+闭定界符"的正则在公式内部含括号时会提前收尾，
//导致公式后半段泄漏成裸 LaTeX（实测 26 条回答里 12 条踩到这个）
const MATH_DELIM=/\\\(|\\\)|\\\[|\\\]|\$\$|\$/g;
const DELIM_CLOSE={"\\(":"\\)","\\[":"\\]","$$":"$$"};

function renderMath(latex,display){
  //模型偶尔把下划线转义成 \_，KaTeX 会当成字面量渲染，这里还原回 _
  const body=String(latex).replace(/\\_/g,"_");
  try{
    return {html:katex.renderToString(body,{displayMode:!!display,throwOnError:false,output:"html"}),
            display:!!display};
  }catch(e){
    return null;
  }
}

//单美元和反斜杠定界符都可能把正文误当公式（例如"成本约 $5 到 $10"），
//先用特征判断内容像不像数学：含中文、含空格但没反斜杠命令的，一律不渲染
function looksLikeMath(body){
  const s=String(body);
  if(!s.trim()){return false;}
  if(/[\u4e00-\u9fa5，。；：！？（）【】]/.test(s)){return false;}
  if(/\\/.test(s)){return true;}
  if(/\s/.test(s)){return false;}
  return /[0-9A-Za-z_^=+\-*/]/.test(s);
}

//按定界符出现顺序配对：不会像正则那样在公式内部误判结束位置
function extractMath(src){
  const map={};
  const text=String(src);
  let out="";
  let cursor=0;
  let i=0;
  MATH_DELIM.lastIndex=0;
  let open=null;
  let inlineUsd=false;   //是否在 "$...$" 里：此时下一个单 $ 就是闭合符
  let match;
  while((match=MATH_DELIM.exec(text))!==null){
    const delim=match[0];
    const pos=match.index;
    if(delim==="$"&&text[pos+1]==="$"){continue;}   //双美元由 $$ 分支处理
    if(open===null){
      if(inlineUsd){
        if(delim!=="$"){continue;}
        const body=text.slice(inlineUsd+1,pos);
        const rendered=looksLikeMath(body)?renderMath(body,false):null;
        if(rendered&&body.trim()){
          out+=text.slice(cursor,inlineUsd);
          const token="@@MATH"+(i++)+"@@";
          map[token]=rendered;
          out+=token;
          cursor=pos+1;
        }
        inlineUsd=false;
        continue;
      }
      const close=DELIM_CLOSE[delim];
      if(!close){
        //单个 $ 在下次遇到 $ 时闭合，且后续必须真的还有 $，否则按普通文本（避免把价格当公式）
        if(delim==="$"&&text.indexOf("$",pos+1)===-1){continue;}
        if(delim==="$"){inlineUsd=pos;}
        continue;
      }
      open={delim,close,start:pos};
      continue;
    }
    if(delim!==open.close){continue;}
    const body=text.slice(open.start+open.delim.length,pos);
    const rendered=looksLikeMath(body)?renderMath(body,open.delim==="\\["||open.delim==="$$"):null;
    if(rendered&&body.trim()){
      //公式之前的普通文本先原样保留
      out+=text.slice(cursor,open.start);
      const token="@@MATH"+(i++)+"@@";
      map[token]=rendered;
      out+=token;
      cursor=pos+delim.length;
    }
    open=null;
  }
  out+=text.slice(cursor);
  return {text:out,map};
}

//单独的块级公式不包 <p>，KaTeX 自己会输出块级元素
const MATH_TOKEN_ONLY=/^\s*@@MATH\d+@@\s*$/;

function inlineMd(s){
  return escapeHtml(String(s))
    .replace(/\*\*([^*]+)\*\*/g,"<strong>$1</strong>")
    .replace(/`([^`]+)`/g,"<code>$1</code>")
    .replace(/&lt;br&gt;/g,"<br>");
}

function restoreMath(html,map){
  return html.replace(/@@MATH(\d+)@@/g,(m,idx)=>(map[m]&&map[m].html)||m);
}

function splitTableRow(s){
  return s.trim().replace(/^\|/,"").replace(/\|$/,"").split("|").map(c=>c.trim());
}

function isTableSep(s){
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(s);
}

function parseAligns(sep){
  return splitTableRow(sep).map((c)=>{
    const left=c.startsWith(":");
    const right=c.endsWith(":");
    return left&&right?"center":right?"right":"left";
  });
}

function buildTable(head,aligns,rows){
  const thead="<tr>"+head.map((c,i)=>`<th style="text-align:${aligns[i]||"left"}">${inlineMd(c)}</th>`).join("")+"</tr>";
  const body=rows.map((r)=>"<tr>"+r.map((c,i)=>`<td style="text-align:${aligns[i]||"left"}">${inlineMd(c)}</td>`).join("")+"</tr>").join("");
  return `<div class="table-wrap"><table><thead>${thead}</thead><tbody>${body}</tbody></table></div>`;
}

function renderMarkdown(src){
  //先把闭合的公式抽成占位符：块级公式可能跨行，按行切会匹配不到
  const extracted=extractMath(src);
  const lines=extracted.text.split("\n");
  let html="";
  let list=null;
  let i=0;
  const closeList=()=>{
    if(list){
      html+=list==="ul"?"</ul>":"</ol>";
      list=null;
    }
  };
  for(;i<lines.length;i++){
    const line=lines[i];
    if(line.includes("|")&&splitTableRow(line).length>1&&i+1<lines.length&&isTableSep(lines[i+1])){
      const head=splitTableRow(line);
      const aligns=parseAligns(lines[i+1]);
      const rows=[];
      let j=i+2;
      while(j<lines.length&&lines[j].includes("|")&&splitTableRow(lines[j]).length>1){
        rows.push(splitTableRow(lines[j]));
        j++;
      }
      closeList();
      html+=buildTable(head,aligns,rows);
      i=j-1;
      continue;
    }
    const head=line.match(/^(#{1,4})\s+(.*)$/);
    if(head){
      closeList();
      const lvl=Math.min(head[1].length+2,6);
      html+=`<h${lvl}>${inlineMd(head[2])}</h${lvl}>`;
      continue;
    }
    const ul=line.match(/^\s*[-*]\s+(.*)$/);
    if(ul){
      if(list!=="ul"){closeList();html+="<ul>";list="ul";}
      html+=`<li>${inlineMd(ul[1])}</li>`;
      continue;
    }
    const ol=line.match(/^\s*\d+[.、]\s+(.*)$/);
    if(ol){
      if(list!=="ol"){closeList();html+="<ol>";list="ol";}
      html+=`<li>${inlineMd(ol[1])}</li>`;
      continue;
    }
    closeList();
    if(!line.trim()){continue;}
    //整行就是一条块级公式时直接输出，不再套段落标签
    const only=line.match(MATH_TOKEN_ONLY);
    if(only&&extracted.map[line.trim()]&&extracted.map[line.trim()].display){
      html+=extracted.map[line.trim()].html;
      continue;
    }
    html+=`<p>${inlineMd(line)}</p>`;
  }
  closeList();
  return restoreMath(html,extracted.map);
}

//同一篇论文可能召回多个片段，展示层按来源合并，避免出现重复引用
function collapseSources(sources){
  const map=new Map();
  (sources||[]).forEach((s)=>{
    const key=(s.source_type||"local")+"||"+(s.paper_id||s.title||s.source||"");
    if(!map.has(key)){
      map.set(key,{head:s,sections:new Set()});
    }
    const entry=map.get(key);
    const section=s.section||"";
    if(section&&!section.startsWith(entry.head.paper_id||entry.head.source||"")){
      entry.sections.add(section);
    }
  });
  return [...map.values()].map((entry)=>{
    const merged={...entry.head};
    merged.matched_sections=[...entry.sections];
    return merged;
  });
}

function renderSources(sources){
  const wrap=document.createElement("div");
  wrap.className="sources";
  collapseSources(sources).forEach((s)=>{
    const chip=document.createElement("div");
    chip.className="source-chip";
    const title=document.createElement("div");
    title.className="source-title";
    title.textContent=s.title||s.source||"未知来源";
    const sec=document.createElement("div");
    sec.className="source-section";
    const type=s.source_type||"local";
    let meta="";
    if(type==="local"){
      //章节可能是解析兜底出来的占位值，只展示真正匹配到的章节
      const paper=s.paper_id||"";
      const extra=(s.matched_sections||[]).join("、");
      meta=paper+(extra?" · "+extra:"");
    }else if(type==="wikipedia"){
      meta="Wikipedia";
    }else if(type==="arxiv"){
      meta="arXiv"+(s.arxiv_id?" "+s.arxiv_id:"");
    }
    sec.textContent=meta;
    const snippet=document.createElement("div");
    snippet.className="source-snippet";
    snippet.textContent=s.snippet||s.text||"";
    if(s.url){
      const link=document.createElement("a");
      link.href=s.url;
      link.target="_blank";
      link.rel="noopener";
      link.textContent="原文链接";
      chip.append(title,sec,snippet,link);
    }else{
      chip.append(title,sec,snippet);
    }
    wrap.append(chip);
  });
  return wrap;
}

$("#login-form").addEventListener("submit",async(e)=>{
  e.preventDefault();
  const err=$("#login-error");
  err.hidden=true;
  const btn=$("#login-btn");
  btn.disabled=true;
  try{
    const resp=await fetch("/login",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({
        username:$("#username").value.trim(),
        password:$("#password").value
      })
    });
    const data=await resp.json().catch(()=>({}));
    if(!resp.ok){
      err.textContent=data.detail||"登录失败";
      err.hidden=false;
      return;
    }
    token=data.access_token;
    sessionStorage.setItem(TOKEN_KEY,token);
    sessionStorage.setItem(USERNAME_KEY,data.username||$("#username").value.trim());
    showChat();
  }catch(ex){
    err.textContent="无法连接服务";
    err.hidden=false;
  }finally{
    btn.disabled=false;
  }
});

$("#logout-btn").addEventListener("click",logout);
$("#new-thread-btn").addEventListener("click",startNewThread);
$("#sidebar-toggle").addEventListener("click",()=>{
  //收起状态记在本地，刷新后保持
  const collapsed=chatView.classList.toggle("sidebar-collapsed");
  localStorage.setItem("paper_agent_sidebar",collapsed?"1":"0");
});
$("#sidebar-show").addEventListener("click",()=>{
  chatView.classList.remove("sidebar-collapsed");
  localStorage.setItem("paper_agent_sidebar","0");
});

async function sendQuestion(){
  if(streaming){return;}
  const question=questionInput.value.trim();
  if(!question){return;}
  //与服务端同一条规则：只有标点或空白的内容不发起请求，省掉一次往返
  if(!/[0-9A-Za-z\u4e00-\u9fa5]/.test(question)){
    questionInput.value="";
    sendBtn.disabled=true;
    const node=addMessage("bot");
    setStatus(node.meta,"请输入具体问题，当前内容只有标点或空白",false);
    node.el.classList.add("has-error");
    return;
  }
  questionInput.value="";
  sendBtn.disabled=true;
  const userNode=addMessage("user",question);
  attachActions(userNode,[
    {cls:"action-copy",icon:"copy",title:"复制",onClick:()=>copyWithFeedback(userNode.el.querySelector(".action-copy"),question)},
    {cls:"action-edit",icon:"pencil",title:"修改并重新提问",onClick:()=>startEditUserMessage(userNode,question)}
  ]);
  const bot=addMessage("bot");
  setStatus(bot.meta,"连接中",true);
  streaming=true;
  try{
    const ok=await streamAnswer(bot,question);
    if(!ok){setStatus(bot.meta,"连接中断",false);}
    attachAnswerActions(bot,question);
  }catch(ex){
    setStatus(bot.meta,"连接中断",false);
  }finally{
    streaming=false;
    sendBtn.disabled=!questionInput.value.trim();
    questionInput.focus();
    refreshThreads();
  }
}

//回答完成后挂上复制与重新生成两个入口
function attachAnswerActions(node,question){
  attachActions(node,[
    {cls:"action-copy-answer",icon:"copy",title:"复制回答",
     onClick:()=>copyWithFeedback(node.el.querySelector(".action-copy-answer"),node.answer||"")},
    {cls:"action-regen",icon:"refresh-cw",title:"重新生成",
     onClick:()=>rerunMessage(node,question)}
  ]);
}

//单一流式实现：首轮提问和重新生成都走这里，避免两套解析逻辑走偏
async function streamAnswer(node,question){
  if(!streaming){streaming=true;}
  node.answer="";
  node.question=question;
  const buffer={text:""};
  let hasError=false;
  const parseEvent=(line)=>{
    if(!line.startsWith("data:")){return;}
    let evt;
    try{evt=JSON.parse(line.slice(5).trim());}catch(ex){return;}
    if(evt.type==="thread"){
      //服务端在首轮对话时创建会话，前端接住ID后刷新侧边栏
      if(evt.thread_id&&evt.thread_id!==activeThread){
        activeThread=evt.thread_id;
        localStorage.setItem(THREAD_KEY,evt.thread_id);
        refreshThreads();
      }
    }else if(evt.type==="route"){
      setRouteBadge(node.meta,evt.route);
    }else if(evt.type==="status"){
      setStatus(node.meta,evt.content,true);
    }else if(evt.type==="progress"){
      //综述链路的分步进度：只展示最新一步，带转圈提示还在跑
      setStatus(node.meta,evt.content,true);
    }else if(evt.type==="ping"){
      //心跳只用来保连接，不改界面
    }else if(evt.type==="sources"){
      setStatus(node.meta,"",false);
      node.el.append(renderSources(evt.sources));
    }else if(evt.type==="token"){
      setStatus(node.meta,"",false);
      node.answer+=evt.content||"";
      node.content.innerHTML=renderMarkdown(node.answer);
      messagesEl.scrollTop=messagesEl.scrollHeight;
    }else if(evt.type==="error"){
      hasError=true;
      node.el.classList.add("has-error");
      setStatus(node.meta,evt.message||"服务异常",false);
    }else if(evt.type==="done"){
      if(!hasError){setStatus(node.meta,"",false);}
    }
  };
  const flushBuffer=()=>{
    let idx;
    while((idx=buffer.text.indexOf("\n\n"))>=0){
      const block=buffer.text.slice(0,idx);
      buffer.text=buffer.text.slice(idx+2);
      block.split("\n").forEach(parseEvent);
    }
  };
  //带上会话ID，后端才会把这条问答挂到同一个会话里
  let url="/ask/stream?question="+encodeURIComponent(question);
  if(activeThread){url+="&thread_id="+encodeURIComponent(activeThread);}
  const resp=await fetch(url,{headers:authHeaders()});
  if(resp.status===401){
    logout();
    return false;
  }
  if(!resp.ok){
    setStatus(node.meta,"请求失败（"+resp.status+"）",false);
    return false;
  }
  const reader=resp.body.getReader();
  const decoder=new TextDecoder();
  while(true){
    const {done,value}=await reader.read();
    if(done){break;}
    buffer.text+=decoder.decode(value,{stream:true});
    flushBuffer();
  }
  flushBuffer();
  return true;
}

sendBtn.addEventListener("click",sendQuestion);
questionInput.addEventListener("input",()=>{
  sendBtn.disabled=!questionInput.value.trim()||streaming;
});
questionInput.addEventListener("keydown",(e)=>{
  if(e.key==="Enter"&&!e.shiftKey){
    e.preventDefault();
    sendQuestion();
  }
});

if(token){showChat();}else{showLogin();}
