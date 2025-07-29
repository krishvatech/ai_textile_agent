from sqlalchemy import Column, Integer, String, ForeignKey, Float, Boolean, DateTime, JSON
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True)
    whatsapp_number = Column(String, unique=True, index=True)
    products = relationship("Product", back_populates="tenant", cascade="all, delete")
    customers = relationship("Customer", back_populates="tenant", cascade="all, delete")
    orders = relationship("Order", back_populates="tenant", cascade="all, delete")

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"))
    name = Column(String)
    description = Column(String)
    color = Column(String)
    price = Column(Float)
    is_rental = Column(Boolean, default=False)
    available_stock = Column(Integer)
    image_url = Column(String)
    metadata = Column(JSON, nullable=True)
    tenant = relationship("Tenant", back_populates="products")

class Customer(Base):
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"))
    whatsapp_id = Column(String, index=True)
    name = Column(String)
    preferred_language = Column(String, default="en")
    tenant = relationship("Tenant", back_populates="customers")
    orders = relationship("Order", back_populates="customer")

class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"))
    customer_id = Column(Integer, ForeignKey("customers.id"))
    product_id = Column(Integer, ForeignKey("products.id"))
    order_type = Column(String)
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    tenant = relationship("Tenant", back_populates="orders")
    customer = relationship("Customer", back_populates="orders")
    product = relationship("Product")
