from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Enum, JSON, Table, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
import enum
Base = declarative_base()
# --- ENUMS ---
class UserRole(str, enum.Enum):
    """Roles for system users (for RBAC and permissions)"""
    superadmin = "superadmin"
    tenant_admin = "tenant_admin"
    staff = "staff"
class OrderStatus(str, enum.Enum):
    """Possible statuses for any order (buy or rental)"""
    placed = "placed"
    paid = "paid"
    shipped = "shipped"
    rented = "rented"
    returned = "returned"
    cancelled = "cancelled"
class RentalStatus(str, enum.Enum):
    """Rental period status (active, returned, cancelled)"""
    active = "active"
    returned = "returned"
    cancelled = "cancelled"
# --- ASSOCIATION TABLES ---
# Bridge: Many-to-many relation between ProductVariant and Occasion
product_variant_occasions = Table(
    'product_variant_occasions', Base.metadata,
    Column('variant_id', Integer, ForeignKey('product_variants.id'), primary_key=True),
    Column('occasion_id', Integer, ForeignKey('occasions.id'), primary_key=True)
)
# --- MAIN TABLES ---
class Tenant(Base):
    """A single shop/business (multi-tenant SaaS support)"""
    __tablename__ = "tenants"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True, nullable=False)
    whatsapp_number = Column(String, unique=True, index=True, nullable=False)
    phone_number = Column(String, nullable=True)
    address = Column(String, nullable=True)
    language = Column(String, default="en")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Relationships
    products = relationship("Product", back_populates="tenant", cascade="all, delete-orphan")
    customers = relationship("Customer", back_populates="tenant", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="tenant", cascade="all, delete-orphan")
    users = relationship("User", back_populates="tenant", cascade="all, delete-orphan")
class User(Base):
    """User (admin, staff) for a tenant/shop. Superadmin is platform-level."""
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.tenant_admin, nullable=False)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)  # Null if superadmin
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Relationships
    tenant = relationship("Tenant", back_populates="users", uselist=False)
class Customer(Base):
    """A customer (buyer/renter) for a specific shop/tenant."""
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    whatsapp_id = Column(String, index=True)  # For WhatsApp-based chatbots
    name = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    preferred_language = Column(String, default="en")
    meta_data = Column(JSON, nullable=True)
    email = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    loyalty_points = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Relationships
    tenant = relationship("Tenant", back_populates="customers")
    orders = relationship("Order", back_populates="customer")
    feedbacks = relationship("Feedback", back_populates="customer", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="customer", cascade="all, delete-orphan")
    __table_args__ = (Index('idx_customers_tenant_id', "tenant_id"),)
class Product(Base):
    """
    Core product definition (e.g., 'Bridal Saree'), common across variants.
    """
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=True)  # e.g., saree, lehenga, sherwani
    category = Column(String, nullable=True)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Relationships
    tenant = relationship("Tenant", back_populates="products")
    variants = relationship("ProductVariant", back_populates="product", cascade="all, delete-orphan")
    images = relationship("ProductImage", back_populates="product", cascade="all, delete-orphan")
class ProductImage(Base):
    """
    Supports multiple images per product (e.g., main, detail, color views).
    """
    __tablename__ = "product_images"
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False, index=True)
    image_url = Column(String, nullable=False)
    is_primary = Column(Boolean, default=False)
    sort_order = Column(Integer, default=0)
    # Relationships
    product = relationship("Product", back_populates="images")
class ProductVariant(Base):
    """
    Represents a unique combination of color, size, and fabric for a product.
    E.g., 'Red, M, Silk' of 'Bridal Saree'.
    """
    __tablename__ = "product_variants"
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False, index=True)
    color = Column(String, nullable=False, index=True)
    size = Column(String, nullable=False, index=True)
    fabric = Column(String, nullable=False, index=True)
    price = Column(Float, nullable=False)
    available_stock = Column(Integer, default=0, nullable=False)
    is_rental = Column(Boolean, default=False, nullable=False)
    rental_price = Column(Float, nullable=True)
    image_url = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Relationships
    product = relationship("Product", back_populates="variants")
    rentals = relationship("Rental", back_populates="product_variant", cascade="all, delete-orphan")
    occasions = relationship("Occasion", secondary=product_variant_occasions, back_populates="variants")
    __table_args__ = (
        UniqueConstraint('product_id', 'color', 'size', 'fabric', name='uix_variant'),
        Index('idx_variant_color_size_fabric', 'color', 'size', 'fabric'),
    )
class Occasion(Base):
    """
    Occasion (wedding, festival, party, etc.)
    Used for filtering/recommending products for specific events.
    """
    __tablename__ = "occasions"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    # Relationships
    variants = relationship("ProductVariant", secondary=product_variant_occasions, back_populates="occasions")
class Rental(Base):
    """
    A single rental event/period for a ProductVariant.
    Used for tracking availability, bookings, and returns.
    """
    __tablename__ = "rentals"
    id = Column(Integer, primary_key=True)
    product_variant_id = Column(Integer, ForeignKey("product_variants.id"), nullable=False, index=True)
    rental_start_date = Column(DateTime, nullable=False)
    rental_end_date = Column(DateTime, nullable=False)
    rental_price = Column(Float, nullable=False)
    status = Column(Enum(RentalStatus), default=RentalStatus.active, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Relationships
    product_variant = relationship("ProductVariant", back_populates="rentals")
class Order(Base):
    """
    Customer order for a buy or rental of a product variant.
    """
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    product_variant_id = Column(Integer, ForeignKey("product_variants.id"), nullable=False)
    order_type = Column(String, default="purchase")  # "purchase" or "rental"
    status = Column(Enum(OrderStatus), default=OrderStatus.placed, nullable=False)
    start_date = Column(DateTime, nullable=True)  # Rental period start (if applicable)
    end_date = Column(DateTime, nullable=True)    # Rental period end (if applicable)
    price = Column(Float, nullable=False)
    payment_status = Column(String, default="pending")  # Or use Enum if preferred
    order_number = Column(String, unique=True, nullable=True)
    comments = Column(String, nullable=True)
    discount = Column(Float, nullable=True)
    meta_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Relationships
    tenant = relationship("Tenant", back_populates="orders")
    customer = relationship("Customer", back_populates="orders")
    product_variant = relationship("ProductVariant")
    __table_args__ = (Index('idx_orders_tenant_id', "tenant_id"),)
# ---- Optional/Future Feature Tables ----
class Feedback(Base):
    """
    Customer feedback or rating for an order.
    """
    __tablename__ = "feedbacks"
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=True)
    rating = Column(Integer, nullable=False)
    comments = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    # Relationships
    customer = relationship("Customer", back_populates="feedbacks")
class ChatSession(Base):
    """
    Customer chat/voice interaction session (with AI bot or human).
    """
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    transcript = Column(JSON, nullable=True)  # List of messages, or plain text
    # Relationships
    customer = relationship("Customer", back_populates="chat_sessions")







