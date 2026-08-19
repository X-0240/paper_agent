const $=(sel)=>document.querySelector(sel);
const loginView=$("#login-view");
const chatView=$("#chat-view");
const messagesEl=$("#messages");
const questionInput=$("#question-input");
const sendBtn=$("#send-btn");
const TOKEN_KEY="paper_agent_token";
const USERNAME_KEY="paper_agent_username";
let token=sessionStorage.getItem(TOKEN_KEY)||"";
let streaming=false;

function showLogin(){
  loginView.hidden=false;
  chatView.hidden=true;
}

function showChat(){
  loginView.hidden=true;
  chatView.hidden=false;
  $("#user-label").textContent=sessionStorage.getItem(USERNAME_KEY)||"";
  questionInput.focus();
  if(window.lucide){lucide.createIcons();}
}

function logout(){
  sessionStorage.removeItem(TOKEN_KEY);
  sessionStorage.removeItem(USERNAME_KEY);
  token="";
  showLogin();
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

function inlineMd(s){
  return escapeHtml(s)
    .replace(/\*\*([^*]+)\*\*/g,"<strong>$1</strong>")
    .replace(/`([^`]+)`/g,"<code>$1</code>");
}

function renderMarkdown(src){
  const lines=src.split("\n");
  let html="";
  let list=null;
  const closeList=()=>{
    if(list){
      html+=list==="ul"?"</ul>":"</ol>";
      list=null;
    }
  };
  for(const line of lines){
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
    html+=`<p>${inlineMd(line)}</p>`;
  }
  closeList();
  return html;
}

function renderSources(sources){
  const wrap=document.createElement("div");
  wrap.className="sources";
  (sources||[]).forEach((s)=>{
    const chip=document.createElement("div");
    chip.className="source-chip";
    const title=document.createElement("div");
    title.className="source-title";
    title.textContent=s.source||"未知来源";
    const sec=document.createElement("div");
    sec.className="source-section";
    sec.textContent=s.section||"";
    const snippet=document.createElement("div");
    snippet.className="source-snippet";
    snippet.textContent=s.text||"";
    chip.append(title,sec,snippet);
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

async function sendQuestion(){
  if(streaming){return;}
  const question=questionInput.value.trim();
  if(!question){return;}
  questionInput.value="";
  sendBtn.disabled=true;
  addMessage("user",question);
  const bot=addMessage("bot");
  setStatus(bot.meta,"连接中",true);
  streaming=true;
  let answer="";
  let hasError=false;
  const buffer={text:""};
  const parseEvent=(line)=>{
    if(!line.startsWith("data:")){return;}
    let evt;
    try{evt=JSON.parse(line.slice(5).trim());}catch(ex){return;}
    if(evt.type==="route"){
      setRouteBadge(bot.meta,evt.route);
    }else if(evt.type==="status"){
      setStatus(bot.meta,evt.content,true);
    }else if(evt.type==="sources"){
      setStatus(bot.meta,"",false);
      bot.el.append(renderSources(evt.sources));
    }else if(evt.type==="token"){
      setStatus(bot.meta,"",false);
      answer+=evt.content||"";
      bot.content.innerHTML=renderMarkdown(answer);
      messagesEl.scrollTop=messagesEl.scrollHeight;
    }else if(evt.type==="error"){
      hasError=true;
      bot.el.classList.add("has-error");
      setStatus(bot.meta,evt.message||"服务异常",false);
    }else if(evt.type==="done"){
      if(!hasError){setStatus(bot.meta,"",false);}
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
  try{
    const resp=await fetch("/ask/stream?question="+encodeURIComponent(question),{
      headers:{Authorization:"Bearer "+token}
    });
    if(resp.status===401){
      logout();
      return;
    }
    if(!resp.ok){
      setStatus(bot.meta,"请求失败（"+resp.status+"）",false);
      return;
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
  }catch(ex){
    setStatus(bot.meta,"连接中断",false);
  }finally{
    streaming=false;
    sendBtn.disabled=!questionInput.value.trim();
    questionInput.focus();
  }
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
