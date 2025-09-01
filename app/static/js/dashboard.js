/* -------------------------------------------
   dashboard.js
   ------------------------------------------- */

/* ---------- THEME: persist + toggle ---------- */
(function initTheme(){
  const saved = localStorage.getItem('theme') || 'dark';
  document.documentElement.setAttribute('data-bs-theme', saved);
  const icon = document.getElementById('themeIcon');
  if (icon) icon.className = saved === 'dark' ? 'bi bi-moon-stars' : 'bi bi-sun';
})();
document.getElementById('themeToggle')?.addEventListener('click', ()=>{
  const html = document.documentElement;
  const current = html.getAttribute('data-bs-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-bs-theme', next);
  localStorage.setItem('theme', next);
  const icon = document.getElementById('themeIcon');
  if (icon) icon.className = next === 'dark' ? 'bi bi-moon-stars' : 'bi bi-sun';
});
// Handle bfcache restores
window.addEventListener("pageshow", e => { if (e.persisted) location.reload(); });

/* ---------- SIDEBAR: mobile toggle ---------- */
(function initSidebarToggle(){
  const btn = document.getElementById('sidebarToggle');
  const sb  = document.getElementById('sidebar');
  if (!btn || !sb) return;
  btn.addEventListener('click', ()=> {
    sb.classList.toggle('open'); // style in CSS: .sidebar.open { transform: translateX(0); }
  });
})();

/* ---------- CHAT: auto-scroll to latest (with images) ---------- */
(function autoScrollChat(){
  const scroller = document.getElementById('chatScroll');
  if (!scroller) return;
  const toBottom = () => { scroller.scrollTop = scroller.scrollHeight; };
  requestAnimationFrame(toBottom);
})();

/* ---------- CHAT: reveal-on-scroll (stagger) ---------- */
(function revealOnScroll(){
  const items = document.querySelectorAll('.msg');
  if (!items.length) return;
  const on = new IntersectionObserver((entries)=>{
    entries.forEach(e=>{
      if (e.isIntersecting){
        e.target.style.willChange = 'transform';
        e.target.classList.add('seen');
        on.unobserve(e.target);
        setTimeout(()=>{ e.target.style.willChange = 'auto'; }, 250);
      }
    });
  }, {root: document.getElementById('chatScroll') || null, threshold: 0.05});
  items.forEach(i=> on.observe(i));
})();

/* ===== Unified, filter-aware table pagination =====
   - Bind a pager with: <nav class="table-pager" data-table=".table-rentals">...</nav>
   - The table should have <tbody> rows (exclude one optional .js-empty row)
   - Filtering code should set `tr.dataset.filtered = '1' | '0'` and then call `pager._repaginate()`
*/
(function initTablePaginators(){
  function attachPaginator(pagerEl){
    // Prevent double-binding
    if (!pagerEl || pagerEl.dataset._bound === "1") return;

    const tableSel = pagerEl.dataset.table || ".table-products";
    const rowSel   = pagerEl.dataset.rowSelector || "tbody > tr";
    const table    = document.querySelector(tableSel);
    const tbody    = table?.querySelector("tbody");
    if (!table || !tbody) { pagerEl.style.display = "none"; return; }

    // Controls
    const btnFirst = pagerEl.querySelector(".tp-first");
    const btnPrev  = pagerEl.querySelector(".tp-prev");
    const btnNext  = pagerEl.querySelector(".tp-next");
    const btnLast  = pagerEl.querySelector(".tp-last");
    const pageEl   = pagerEl.querySelector(".tp-page");
    const pagesEl  = pagerEl.querySelector(".tp-pages");
    const sizeSel  = pagerEl.querySelector(".tp-page-size");

    let pageSize = parseInt(sizeSel?.value || "20", 10) || 20;
    let page     = 1;

    // Helpers
    const getAllRows = () =>
      Array.from(tbody.querySelectorAll(rowSel))
        .filter(r => !r.classList.contains("js-empty")); // ignore empty-state row

    const getFilteredRows = () =>
      getAllRows().filter(r => (r.dataset.filtered ?? "1") !== "0");

    function render(){
      const all  = getAllRows();
      const rows = getFilteredRows();
      const count = rows.length;

      const totalPages = Math.max(1, Math.ceil(count / pageSize));
      if (page > totalPages) page = totalPages;

      // Hide all, then show the current slice
      all.forEach(r => { r.style.display = "none"; });
      const start = (page - 1) * pageSize;
      const end   = start + pageSize;
      rows.slice(start, end).forEach(r => { r.style.display = ""; });

      // Update UI
      if (pageEl)  pageEl.textContent  = String(page);
      if (pagesEl) pagesEl.textContent = String(totalPages);

      const atStart = page <= 1, atEnd = page >= totalPages;
      if (btnFirst) btnFirst.disabled = atStart;
      if (btnPrev)  btnPrev.disabled  = atStart;
      if (btnNext)  btnNext.disabled  = atEnd;
      if (btnLast)  btnLast.disabled  = atEnd;
      const autohide = pagerEl.dataset.autohide !== "0";
      pagerEl.style.display = (count === 0 || (autohide && count <= pageSize)) ? "none" : "";
      // // Auto-hide pager if not needed
      // pagerEl.style.display = (count === 0 || count <= pageSize) ? "none" : "";
    }

    // Events
    btnFirst?.addEventListener("click", () => { page = 1; render(); });
    btnPrev ?.addEventListener("click", () => { if (page > 1) page--; render(); });
    btnNext ?.addEventListener("click", () => {
      const tp = Math.max(1, Math.ceil(getFilteredRows().length / pageSize));
      if (page < tp) page++; render();
    });
    btnLast ?.addEventListener("click", () => {
      page = Math.max(1, Math.ceil(getFilteredRows().length / pageSize));
      render();
    });
    sizeSel ?.addEventListener("change", () => {
      pageSize = parseInt(sizeSel.value || "20", 10) || 20;
      page = 1; render();
    });

    // Public hook for tab filters (call this after toggling data-filtered)
    pagerEl._repaginate = () => { page = 1; render(); };

    // Observe row additions/removals and re-render
    const mo = new MutationObserver(() => { page = 1; render(); });
    mo.observe(tbody, { childList: true });

    // Initial paint & mark bound
    render();
    pagerEl.dataset._bound = "1";
    console.log(`[pager] ${tableSel}: ${getAllRows().length} total rows (filtered now: ${getFilteredRows().length})`);
  }

  function initAll(){
    document.querySelectorAll(".table-pager").forEach(attachPaginator);
  }

  if (document.readyState === "loading"){
    document.addEventListener("DOMContentLoaded", initAll);
  } else {
    initAll();
  }
})();
