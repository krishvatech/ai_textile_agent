from fastapi import Depends, HTTPException, status, Header
from app.models import User, UserRole
from typing import Optional

# --- Dummy user extraction for demo. Replace with real auth logic (JWT/session/DB)
# In production, you'd decode a JWT or session to get the user.

# Simulated in-memory users (demo only)
fake_users_db = {
    "admin@example.com": User(id=1, email="admin@example.com", role=UserRole.superadmin, is_active=True),
    "shop1@example.com": User(id=2, email="shop1@example.com", role=UserRole.tenant_admin, tenant_id=1, is_active=True),
    "staff1@example.com": User(id=3, email="staff1@example.com", role=UserRole.staff, tenant_id=1, is_active=True)
}

def get_current_user(x_user_email: Optional[str] = Header(None)):
    """
    Demo: Reads X-User-Email header to get the user.
    Replace this with your real JWT/session logic.
    """
    if not x_user_email or x_user_email not in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing user credentials"
        )
    user = fake_users_db[x_user_email]
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Inactive user")
    return user

def require_superadmin(user: User = Depends(get_current_user)):
    if user.role != UserRole.superadmin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superadmin only"
        )
    return user

def require_tenant_admin(user: User = Depends(get_current_user)):
    if user.role not in [UserRole.tenant_admin, UserRole.superadmin]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin only"
        )
    return user

def require_staff_or_admin(user: User = Depends(get_current_user)):
    if user.role not in [UserRole.staff, UserRole.tenant_admin, UserRole.superadmin]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Staff or admin only"
        )
    return user
