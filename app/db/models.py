from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Enum, JSON, Index
)
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
import enum

Base = declarative_base()

class UserRole(str, enum.Enum):
    superadmin = "superadmin"
    tenant_admin = "tenant_admin"
    staff = "staff"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.tenant_admin)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)  # Null if superadmin
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    tenant = relationship("Tenant", back_populates="users", uselist=False)
    
# --- Order Status Enum
class OrderStatus(str, enum.Enum):
    placed = "placed"
    paid = "paid"
    shipped = "shipped"
    rented = "rented"
    returned = "returned"
    cancelled = "cancelled"

# --- Tenant (Shop/Business)
class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True)
    whatsapp_number = Column(String, unique=True, index=True)
    phone_number = Column(String, nullable=True)
    address = Column(String, nullable=True)
    language = Column(String, default="en")  # Default language for this tenant
    products = relationship("Product", back_populates="tenant", cascade="all, delete-orphan")
    customers = relationship("Customer", back_populates="tenant", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="tenant", cascade="all, delete-orphan")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    users = relationship("User", back_populates="tenant", cascade="all, delete-orphan")

# --- Product (Multi-language)
# class Product(Base):
#     __tablename__ = "products"
#     id = Column(Integer, primary_key=True)
#     tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
#     name_en = Column(String, nullable=False)
#     name_hi = Column(String, nullable=True)
#     name_gu = Column(String, nullable=True)
#     description_en = Column(String, nullable=True)
#     description_hi = Column(String, nullable=True)
#     description_gu = Column(String, nullable=True)
#     color = Column(String, nullable=True)
#     type = Column(String, nullable=True)  # Chaniya choli, saree, sherwani, etc.
#     price = Column(Float, nullable=False)
#     is_rental = Column(Boolean, default=False)
#     available_stock = Column(Integer, default=0)
#     image_url = Column(String, nullable=True)
#     meta_data = Column(JSON, nullable=True)
#     sku = Column(String, unique=True, nullable=True)
#     rental_price = Column(Float, nullable=True)
#     category = Column(String, nullable=True)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
#     is_active = Column(Boolean, default=True)
#     tenant = relationship("Tenant", back_populates="products")

#     __table_args__ = (Index('idx_products_tenant_id', "tenant_id"),)

# --- Customer (Per-tenant)
class Customer(Base):
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    whatsapp_id = Column(String, index=True)
    name = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    preferred_language = Column(String, default="en")
    meta_data = Column(JSON, nullable=True)
    tenant = relationship("Tenant", back_populates="customers")
    orders = relationship("Order", back_populates="customer")
    email = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    __table_args__ = (Index('idx_customers_tenant_id', "tenant_id"),)

class OrderStatus(str, enum.Enum):
    placed = "placed"
    paid = "paid"
    shipped = "shipped"
    rented = "rented"
    returned = "returned"
    cancelled = "cancelled"
    
# --- Order (Rental or Purchase)
class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    order_type = Column(String, default="purchase")  # or "rental"
    status = Column(Enum(OrderStatus), default=OrderStatus.placed)
    start_date = Column(DateTime, nullable=True)  # For rental
    end_date = Column(DateTime, nullable=True)    # For rental
    price = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    meta_data = Column(JSON, nullable=True)
    tenant = relationship("Tenant", back_populates="orders")
    customer = relationship("Customer", back_populates="orders")
    product = relationship("Product")
    payment_status = Column(String, default="pending")  # Or as Enum
    order_number = Column(String, unique=True, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    comments = Column(String, nullable=True)
    discount = Column(Float, nullable=True)

    __table_args__ = (Index('idx_orders_tenant_id', "tenant_id"),)


class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    sku = Column(String, unique=True, nullable=True)
    name_en = Column(String, nullable=False)
    name_hi = Column(String, nullable=True)
    name_gu = Column(String, nullable=True)
    description_en = Column(String, nullable=True)
    description_hi = Column(String, nullable=True)
    description_gu = Column(String, nullable=True)
    color = Column(String, nullable=True)
    type = Column(String, nullable=True)
    category = Column(String, nullable=True)
    price = Column(Float, nullable=False)
    rental_price = Column(Float, nullable=True)
    is_rental = Column(Boolean, default=False)
    available_stock = Column(Integer, default=0)
    image_url = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    meta_data = Column(JSON, nullable=True)
    tenant = relationship("Tenant", back_populates="products")
    __table_args__ = (Index('idx_products_tenant_id', "tenant_id"),)
