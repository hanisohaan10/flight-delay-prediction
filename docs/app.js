/* ---- CONFIG ---------------------------------------------------------- */
const CONFIG = {
  // Paste your Google OAuth Web client ID here to enable "Sign in with Google".
  // Leave empty and the site still works fully — sign-in is optional.
  googleClientId: ""
};

const BASE_RATE = 0.18;
const DOW = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"];
const BANDS = [
  [0.15, "Low",      "#5B6E2E", "below the typical flight"],
  [0.25, "Typical",  "#7C7A33", "about average for U.S. flights"],
  [0.40, "Elevated", "#B07A2B", "higher than the average flight"],
  [1.01, "High",     "#9E4324", "well above the average flight"]
];
const POS = "#B07A2B", NEG = "#586A2C";

let MODEL, ENC, ROUTES, ORIGIN_DESTS, CITY, AIRLINES, INSIGHTS;

const $ = id => document.getElementById(id);

/* ---- load assets ----------------------------------------------------- */
async function boot(){
  const j = f => fetch("assets/"+f).then(r=>r.json());
  [MODEL, ENC, ROUTES, ORIGIN_DESTS, CITY, AIRLINES, INSIGHTS] = await Promise.all([
    j("model.json"), j("encoders.json"), j("route_distance.json"),
    j("origin_dests.json"), j("airport_city.json"), j("airlines.json"), j("insights.json")
  ]);
  fillInputs();
  fillInsights();
  $("predict").addEventListener("click", predict);
  initAuth();
}

/* ---- inputs ---------------------------------------------------------- */
function airportLabel(code){ return CITY[code] ? code+" — "+CITY[code] : code; }

function fillInputs(){
  const al = $("airline");
  Object.entries(AIRLINES).sort((a,b)=>a[1].localeCompare(b[1]))
    .forEach(([c,n])=> al.add(new Option(n+" ("+c+")", c)));

  const origins = Object.keys(ORIGIN_DESTS).sort();
  const o = $("origin");
  origins.forEach(c=> o.add(new Option(airportLabel(c), c)));
  o.value = origins.includes("ATL") ? "ATL" : origins[0];

  const dow = $("dow");
  DOW.forEach((d,i)=> dow.add(new Option(d, i+1)));

  const hour = $("hour");
  for(let h=0; h<24; h++){
    const s = h<12?"AM":"PM", h12=(h%12)||12;
    hour.add(new Option(String(h).padStart(2,"0")+":00  ("+h12+" "+s+")", h));
  }
  hour.value = 8;

  o.addEventListener("change", fillDests);
  $("dest").addEventListener("change", showDist);
  fillDests();
}

function fillDests(){
  const o = $("origin").value, dest = $("dest");
  dest.innerHTML = "";
  (ORIGIN_DESTS[o]||[]).forEach(c=> dest.add(new Option(airportLabel(c), c)));
  showDist();
}

function showDist(){
  const d = ROUTES[$("origin").value+"-"+$("dest").value];
  $("dist").innerHTML = d!=null ? "Route distance (from schedule): <b>"+d.toLocaleString()+" miles</b>" : "";
}

/* ---- model inference (matches XGBoost predict_proba) ----------------- */
function featureVector(){
  const al=$("airline").value, o=$("origin").value, d=$("dest").value;
  const dist = ROUTES[o+"-"+d];
  const gm = ENC.global_mean;
  return [ +$("dow").value, +$("hour").value, dist,
    ENC.airline_delay_rate[al] ?? gm,
    ENC.origin_delay_rate[o]  ?? gm,
    ENC.dest_delay_rate[d]    ?? gm ];
}

function infer(x){
  let m = MODEL.base_margin;
  const contrib = new Float64Array(6);
  for(const t of MODEL.trees){
    let i = 0;
    while(t[i][0] !== -1){
      const node = t[i], f = node[0];
      const c = (Math.fround(x[f]) < Math.fround(node[1])) ? node[2] : node[3];
      contrib[f] += t[c][4] - node[4];
      i = c;
    }
    m += t[i][1];
  }
  return { p: 1/(1+Math.exp(-m)), contrib };
}

function bandFor(p){ for(const b of BANDS) if(p<b[0]) return b; return BANDS[BANDS.length-1]; }

/* ---- predict + render ------------------------------------------------ */
function predict(){
  const o=$("origin").value, d=$("dest").value, al=$("airline").value;
  const dist = ROUTES[o+"-"+d];
  if(dist==null){ return; }
  const x = featureVector();
  const { p, contrib } = infer(x);
  const [ , name, color, note ] = bandFor(p);
  const rel = p/BASE_RATE;
  const hr = +$("hour").value, hs = ((hr%12)||12)+" "+(hr<12?"AM":"PM");
  const oneIn = Math.max(2, Math.round(1/p));

  const C = 2*Math.PI*80, dash = Math.min(p,1)*C;
  const labels = [
    [DOW[(+$("dow").value)-1], contrib[0]],
    [hs+" departure", contrib[1]],
    [dist.toLocaleString()+" mi flight", contrib[2]],
    [AIRLINES[al]+" history", contrib[3]],
    [o+" origin", contrib[4]],
    [d+" destination", contrib[5]]
  ].sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));
  const maxAbs = Math.max(...labels.map(l=>Math.abs(l[1]))) || 1;

  const bars = labels.map(([lab,v])=>{
    const w = Math.round(Math.abs(v)/maxAbs*46);
    const fill = v>=0
      ? '<div class="fill" style="left:50%;width:'+w+'%;background:'+POS+'"></div>'
      : '<div class="fill" style="right:50%;width:'+w+'%;background:'+NEG+'"></div>';
    return '<div class="bar-row"><div class="top"><b>'+lab+'</b><span>'+(v>=0?"raises risk":"lowers risk")+
      '</span></div><div class="track"><div class="center"></div>'+fill+'</div></div>';
  }).join("");

  const res = $("result");
  res.className = "result";
  res.innerHTML =
    '<div class="gauge-wrap"><svg viewBox="0 0 200 200">'+
      '<circle cx="100" cy="100" r="80" fill="none" stroke="#EAE3C9" stroke-width="16"/>'+
      '<circle cx="100" cy="100" r="80" fill="none" stroke="'+color+'" stroke-width="16" stroke-linecap="round" '+
        'stroke-dasharray="'+dash+' '+C+'" transform="rotate(-90 100 100)"/>'+
    '</svg><div class="gauge-num"><div class="pct">'+Math.round(p*100)+'%</div>'+
      '<div class="band" style="color:'+color+'">'+name+' risk</div></div></div>'+
    '<div class="verdict">Roughly a <b>1-in-'+oneIn+'</b> chance this flight leaves 15+ minutes late — '+note+'.</div>'+
    '<div class="rel-pill" style="background:'+color+'22;color:'+color+'">'+rel.toFixed(1)+'× the average flight</div>'+
    '<div class="drivers"><div class="dh">What\'s driving this</div>'+bars+
      '<div class="legend"><span class="sw" style="background:'+POS+'"></span>raises risk'+
      '<span class="sw" style="background:'+NEG+'"></span>lowers risk</div></div>';
}

/* ---- insights -------------------------------------------------------- */
function fillInsights(){
  const s = INSIGHTS.stats;
  $("c-flights").textContent = (s.flights/1e6).toFixed(1)+"M";
  $("s-base").textContent = Math.round(s.base_rate*100)+"%";
  const peak = INSIGHTS.hour.reduce((a,b)=>b.rate>a.rate?b:a);
  $("s-peak").textContent = Math.round(peak.rate*100)+"%";

  const ri = (x)=> '<div class="ri"><span>'+x.code+' &middot; '+x.city+'</span><span class="r">'+Math.round(x.rate*100)+'%</span></div>';
  $("worst").innerHTML = INSIGHTS.worst.map(ri).join("");
  $("best").innerHTML  = INSIGHTS.best.map(ri).join("");

  new Chart($("hourChart"), {
    type:"line",
    data:{ labels: INSIGHTS.hour.map(h=> (((h.h%12)||12))+(h.h<12?"a":"p")),
      datasets:[{ data: INSIGHTS.hour.map(h=>+(h.rate*100).toFixed(1)),
        borderColor:"#586A2C", backgroundColor:"rgba(88,106,44,.12)",
        fill:true, tension:.4, pointRadius:0, borderWidth:3 }] },
    options:{ responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{display:false}, tooltip:{callbacks:{label:c=>c.parsed.y+"% delayed"}} },
      scales:{ y:{ ticks:{callback:v=>v+"%",color:"#6f6e54"}, grid:{color:"#EAE3C9"}, beginAtZero:true },
               x:{ ticks:{color:"#6f6e54",maxRotation:0,autoSkip:true,maxTicksLimit:10}, grid:{display:false} } } }
  });
}

/* ---- nav + reveal ---------------------------------------------------- */
const nav = $("nav");
const hero = document.querySelector(".hero");
window.addEventListener("scroll", ()=>{
  nav.classList.toggle("scrolled", scrollY>40);
  nav.classList.toggle("on-hero", scrollY < hero.offsetHeight-80);
});
const io = new IntersectionObserver((es)=>es.forEach(e=>{ if(e.isIntersecting){ e.target.classList.add("in"); io.unobserve(e.target);} }),{threshold:.12});
document.querySelectorAll(".reveal").forEach(el=>io.observe(el));

/* ---- Google sign-in (optional) --------------------------------------- */
function initAuth(){
  const btn = $("gbtn");
  if(!CONFIG.googleClientId){
    btn.addEventListener("click", ()=> $("modal").classList.add("open"));
    return;
  }
  const s = document.createElement("script");
  s.src = "https://accounts.google.com/gsi/client"; s.async = true;
  s.onload = ()=>{
    google.accounts.id.initialize({ client_id: CONFIG.googleClientId, callback: onCredential });
    btn.addEventListener("click", ()=> google.accounts.id.prompt());
  };
  document.head.appendChild(s);
}
function onCredential(resp){
  try{
    const p = JSON.parse(atob(resp.credential.split(".")[1]));
    const btn = $("gbtn");
    btn.innerHTML = (p.picture? '<img src="'+p.picture+'" style="width:22px;height:22px;border-radius:50%">':'')+
      ' '+(p.given_name||p.name||"Signed in");
  }catch(e){}
}

document.addEventListener("DOMContentLoaded", boot);
