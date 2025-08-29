// Email regex and elements
const emailPattern = /[^@\s]+@[^@\s]+\.[^@\s]+/;
const form = document.getElementById('loginForm');
const email = document.getElementById('email');
const password = document.getElementById('password');
const togglePass = document.getElementById('togglePass');
const strengthBar = document.getElementById('strengthBar');
const loginBtn = document.getElementById('loginBtn');

// Card tilt
const card = document.querySelector('.login-card');
let bounds = null;
const clamp = (n, min, max) => Math.min(Math.max(n, min), max);
if (card) {
  card.addEventListener('mousemove', (e)=>{
    bounds = bounds || card.getBoundingClientRect();
    const x = (e.clientX - bounds.left) / bounds.width;
    const y = (e.clientY - bounds.top) / bounds.height;
    const rx = (y - .5) * 8, ry = (x - .5) * -10;
    card.style.transform = `perspective(900px) rotateX(${rx}deg) rotateY(${ry}deg)`;
  });
  card.addEventListener('mouseleave', ()=>{
    card.style.transform = 'none';
    bounds = null;
  });
}

// Show/Hide password
togglePass?.addEventListener('click', ()=>{
  const isPwd = password.type === 'password';
  password.type = isPwd ? 'text' : 'password';
  togglePass.innerHTML = isPwd ? '<i class="bi bi-eye-slash"></i>' : '<i class="bi bi-eye"></i>';
  password.focus();
});

// Password strength
function strength(v){
  let score = 0;
  if(v.length >= 6) score += 20;
  if(/[A-Z]/.test(v)) score += 20;
  if(/[0-9]/.test(v)) score += 20;
  if(/[^A-Za-z0-9]/.test(v)) score += 20;
  if(v.length >= 10) score += 20;
  return clamp(score, 0, 100);
}
password?.addEventListener('input', (e)=>{
  if (!strengthBar) return;
  strengthBar.style.width = strength(e.target.value) + '%';
});

// Light client validation + spinner (do NOT block normal POST submit)
form?.addEventListener('submit', ()=>{
  let ok = true;
  if(!emailPattern.test(email.value)){ email.classList.add('is-invalid'); ok = false; } else { email.classList.remove('is-invalid'); }
  if(password.value.length < 6){ password.classList.add('is-invalid'); ok = false; } else { password.classList.remove('is-invalid'); }
  if(!ok){ return; }
  const spinner = loginBtn.querySelector('.spinner-border');
  const label = loginBtn.querySelector('.btn-label');
  spinner.classList.remove('d-none');
  label.textContent = 'Signing in…';
  loginBtn.disabled = true;
  // Let the browser submit the form normally to /login
});

window.addEventListener('pageshow', function (e) {
    if (e.persisted) {
      // Force a fresh request so server can bounce to /dashboard if logged in
      location.reload();
    }
  });