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

/* ---------- Demo data (replace with fetch to your FastAPI) ---------- */
const tenant = { id: 4, name: "Vastrang Textile", language: "en" };
const customers = [
  {id:27, name:"Aarti Patel", phone:"917410852078", whatsapp_id:"917410852078", loyalty_points:140, is_active:true, updated_at:"2025-08-27"},
  {id:28, name:"Rahul Shah", phone:"919714452470", whatsapp_id:"919714452470", loyalty_points:60, is_active:true, updated_at:"2025-08-26"},
];
const productCards = [
  {id:101, name:"Beige silk saree with contrast border", type:"Saree", category:"bridal", fabric:"Silk", color:"Beige", size:"Freesize", price:4199, rental_price:799, available_stock:12, image_url:"https://pictures.kartmax.in/cover/live/600x800/quality=6/sites/9s145MyZrWdIAwpU0JYS/product-images/maroon_beige_tissue_silk_saree_with_heavy_blouse_175585771734398_maroon_9.jpg"},
  {id:102, name:"Emerald lehenga set", type:"Lehenga", category:"festival", fabric:"Georgette", color:"Green", size:"M", price:11999, rental_price:2499, available_stock:5, image_url:"https://pictures.kartmax.in/cover/live/600x800/quality=6/sites/9s145MyZrWdIAwpU0JYS/product-images/maroon_beige_tissue_silk_saree_with_heavy_blouse_175585771734398_maroon_9.jpg"},
];
const orders = [
  {id:5001,type:"purchase",customer:"Aarti Patel",variant:"#101 Beige Silk (Freesize)",status:"placed",price:4199,start:null,end:null},
  {id:5002,type:"rental",customer:"Rahul Shah",variant:"#102 Emerald Lehenga (M)",status:"shipped",price:2499,start:"2025-08-23",end:"2025-08-26"},
];
const sessions = [
  {id:9001, customer:"Aarti Patel", started:"2025-08-26 09:07", ended:"2025-08-26 09:14", messages:12},
];

/* ---------- Helpers ---------- */
const toastInst = document.getElementById('liveToast')
  ? new bootstrap.Toast(document.getElementById('liveToast'))
  : null;
const showToast = (msg="Saved!") => {
  const el = document.getElementById('toastMsg');
  if (el) el.textContent = msg;
  toastInst?.show();
};
const rupee = x => new Intl.NumberFormat('en-IN',{style:'currency',currency:'INR',maximumFractionDigits:0}).format(x);

/* ---------- Render single-tenant context ---------- */
function initTenant(){
  const setText = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  setText('tenantName', tenant.name);
  setText('tenantName2', tenant.name);
  setText('sbTenantName', tenant.name);
  setText('sbTenantId', tenant.id);
  setText('tenantLang', tenant.language);
}

/* KPI demo (replace with API aggregates) */
function renderKpis(){
  const setText = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  setText('kpiCustomers', customers.length.toLocaleString('en-IN'));
  setText('kpiOrders', orders.length.toLocaleString('en-IN'));
  setText('kpiRentals', orders.filter(o=>o.type==='rental').length.toLocaleString('en-IN'));
  setText('kpiSkus', productCards.length.toLocaleString('en-IN'));
  const rev = orders.reduce((s,o)=>s+o.price,0).toLocaleString('en-IN');
  const revEl = document.getElementById('revenueHint'); if (revEl) revEl.textContent = rev;
}

/* Tables/Cards */
function renderCustomers(list=customers){
  const tbody = document.getElementById('customerRows');
  if (!tbody) return;
  tbody.innerHTML = list.map(c=>`
    <tr>
      <td>${c.id}</td>
      <td>${c.name}</td>
      <td>${c.phone??'-'}</td>
      <td>${c.whatsapp_id??'-'}</td>
      <td>${c.loyalty_points}</td>
      <td>${c.is_active?'<span class="badge text-bg-success">Yes</span>':'<span class="badge text-bg-secondary">No</span>'}</td>
      <td>${c.updated_at}</td>
      <td class="text-end"><button class="btn btn-sm btn-outline-secondary"><i class="bi bi-eye"></i></button></td>
    </tr>
  `).join('');
}
function renderProducts(){
  const grid = document.getElementById('productGrid');
  if (!grid) return;
  grid.classList.remove("row","g-3");
  grid.innerHTML = productCards.map(p=>`
    <div class="card glass border-0 mb-3">
      <div class="card-body d-flex gap-3 align-items-center">
        <img src="${p.image_url}" alt="${p.name}"
             style="width:140px; height:100px; object-fit:cover; border-radius:var(--radius);">
        <div class="flex-grow-1">
          <h6 class="mb-1 fw-semibold">${p.name} <span class="chip">${p.type}</span></h6>
          <div class="text-secondary small mb-1">
            Fabric: ${p.fabric} • Color: ${p.color} • Size: ${p.size}
          </div>
          <div class="d-flex flex-wrap gap-3 align-items-center">
            <span class="fw-bold">${rupee(p.price)}</span>
            ${p.rental_price ? `<span class="text-secondary small">Rent ${rupee(p.rental_price)}</span>` : ""}
            <span class="${p.available_stock>0?'text-success':'text-danger'} small">
              ${p.available_stock>0?(`${p.available_stock} in stock`):'Out of stock'}
            </span>
          </div>
        </div>
        <div class="ms-auto d-flex flex-column gap-2">
          <button class="btn btn-sm btn-outline-primary"><i class="bi bi-pencil"></i> Edit</button>
          <button class="btn btn-sm btn-outline-danger"><i class="bi bi-trash"></i> Delete</button>
        </div>
      </div>
    </div>
  `).join('');
}
function renderOrders(){
  const tbody = document.getElementById('orderRows');
  if (!tbody) return;
  tbody.innerHTML = orders.map(o=>`
    <tr>
      <td>${o.id}</td>
      <td>${o.type==='rental'?'<span class="badge text-bg-info">Rental</span>':'<span class="badge text-bg-primary">Purchase</span>'}</td>
      <td>${o.customer}</td>
      <td>${o.variant}</td>
      <td><span class="badge text-bg-${o.status==='delivered'?'success':(o.status==='shipped'?'warning':'secondary')} text-dark">${o.status}</span></td>
      <td>${rupee(o.price)}</td>
      <td>${o.start??'-'}</td>
      <td>${o.end??'-'}</td>
      <td class="text-end"><button class="btn btn-sm btn-outline-secondary"><i class="bi bi-eye"></i></button></td>
    </tr>
  `).join('');
}
function renderSessions(){
  const tbody = document.getElementById('sessionRows');
  if (!tbody) return;
  tbody.innerHTML = sessions.map(s=>`
    <tr>
      <td>${s.id}</td>
      <td>${s.customer}</td>
      <td>${s.started}</td>
      <td>${s.ended??'<span class="text-warning">— live —</span>'}</td>
      <td>${s.messages}</td>
      <td class="text-end"><button class="btn btn-sm btn-outline-secondary"><i class="bi bi-chat-right-text"></i></button></td>
    </tr>
  `).join('');
}

/* Search customers */
document.getElementById('customerSearch')?.addEventListener('input', (e)=>{
  const q = e.target.value.toLowerCase();
  const filtered = customers.filter(c => (c.name.toLowerCase().includes(q) || (c.phone??'').includes(q)));
  renderCustomers(filtered);
});

/* Settings */
document.getElementById('saveSettings')?.addEventListener('click', ()=>{
  const sel = document.getElementById('defaultLang');
  if (!sel) return;
  tenant.language = sel.value;
  const lang = document.getElementById('tenantLang');
  if (lang) lang.textContent = tenant.language;
  showToast('Settings saved');
});

/* ---------- SIMPLE HASH ROUTER (shows only active tab) ---------- */
function showSection(id, pushHash=true){
  document.querySelectorAll('#sidebarNav .nav-link').forEach(l=>l.classList.remove('active'));
  const activeLink = document.querySelector(`#sidebarNav .nav-link[data-target="${id}"]`);
  if(activeLink) activeLink.classList.add('active');

  document.querySelectorAll('.page-section').forEach(sec=>sec.classList.add('d-none'));
  const target = document.getElementById(id);
  if(target){ target.classList.remove('d-none'); }

  if(pushHash){
    if(history.replaceState){
      history.replaceState(null, "", `#${id}`);
    }else{
      location.hash = `#${id}`;
    }
  }
}
document.querySelectorAll('#sidebarNav .nav-link').forEach(link=>{
  link.addEventListener('click', (e)=>{
    e.preventDefault();
    const id = link.dataset.target;
    showSection(id, true);
  });
});
function handleInitialRoute(){
  const idFromHash = (location.hash || '#overview').replace('#','');
  const valid = ['overview','customers','products','orders','sessions','settings'];
  showSection(valid.includes(idFromHash) ? idFromHash : 'overview', false);
}

/* ---------- Prevent leaving /dashboard but allow tab Back/Forward ---------- */
(function(){
  const basePath = location.pathname; // e.g., "/dashboard"
  // Seed the history entry (keeps current tab in state too)
  history.replaceState({lock:true, tab:(location.hash || '#overview').slice(1)}, "", location.href);

  window.addEventListener('popstate', function(){
    // If path tries to change (e.g., to /login), block it
    if (location.pathname !== basePath) {
      history.pushState({lock:true, tab:(location.hash || '#overview').slice(1)}, "", location.href);
      return;
    }
    // Inside dashboard → sync UI with current hash
    const id = (location.hash || '#overview').slice(1);
    showSection(VALID_TABS.includes(id) ? id : 'overview', false);
  });

  // Fallback for code that changes only the hash (not pushState)
  window.addEventListener('hashchange', function(){
    const id = (location.hash || '#overview').slice(1);
    showSection(VALID_TABS.includes(id) ? id : 'overview', false);
  });
})();

