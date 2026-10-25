const $ = (id) => document.getElementById(id);
let current = null, tab = "overlay";

function showError(msg) { const e = $("error"); e.textContent = msg; e.hidden = !msg; }

function showResult(r) {
  current = r;
  $("result").hidden = false;
  const forged = !!r.forged;
  const b = $("badge");
  b.textContent = forged ? "FORGED" : "AUTHENTIC";
  b.className = "badge " + (forged ? "forged" : "ok");
  $("m-conf").textContent = Math.round(r.confidence * 100) + "%";
  $("m-score").textContent = r.score.toFixed(3);
  $("m-reg").textContent = r.regions;
  $("m-sec").textContent = (r.seconds || 0).toFixed(1) + " s";
  $("note").textContent = forged
    ? "Possible edits are outlined in red on the first tab. Check them by eye against the original document."
    : "No edit detected. This does not prove the document is genuine: about 3 in 10 computer-made forgeries are missed.";
  setTab(images(r).overlay ? "overlay" : "original");
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}
function images(r) { return r.images || {}; }
function setTab(k) {
  tab = k;
  document.querySelectorAll("#tabs button").forEach((b) => {
    b.classList.toggle("on", b.dataset.k === k);
    b.hidden = !images(current)[b.dataset.k];
  });
  $("view").src = images(current)[k] || "";
}
$("tabs").addEventListener("click", (e) => { if (e.target.dataset.k) setTab(e.target.dataset.k); });

async function upload(file) {
  showError("");
  if (!file) return;
  $("busy").hidden = false;
  const fd = new FormData(); fd.append("file", file);
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");
    showResult(data);
    loadHistory();
  } catch (e) {
    showError(e.message || "Something went wrong.");
  } finally {
    $("busy").hidden = true;
    $("file").value = "";
  }
}

const drop = $("drop");
drop.addEventListener("click", () => $("file").click());
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") $("file").click(); });
$("file").addEventListener("change", (e) => upload(e.target.files[0]));
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));

function esc(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

async function loadHistory() {
  const data = await (await fetch("/api/history")).json();
  const tb = document.querySelector("#hist tbody");
  tb.innerHTML = "";
  $("empty").hidden = data.items.length > 0;
  data.items.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.id}</td><td>${esc(r.created)}</td><td class="f">${esc(r.filename)}</td>
      <td><span class="tag ${r.forged ? "forged" : "ok"}">${esc(r.label)}</span></td>
      <td>${Math.round(r.confidence * 100)}%</td><td>${r.regions}</td>
      <td><button class="del" title="Delete">&times;</button></td>`;
    tr.addEventListener("click", () => showResult(r));
    tr.querySelector(".del").addEventListener("click", async (e) => {
      e.stopPropagation();
      await fetch("/api/history/" + r.id, { method: "DELETE" });
      loadHistory();
    });
    tb.appendChild(tr);
  });
  $("stats").textContent = `${data.stats.total} checked · ${data.stats.forged} flagged`;
}
loadHistory();
