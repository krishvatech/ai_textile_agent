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
window.addEventListener("pageshow", e => { if (e.persisted) location.reload(); });
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

