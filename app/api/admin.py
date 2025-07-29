from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from app.db.session import get_db
from app.db.models import Product, Order
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
import io

router = APIRouter()

@router.get("/products/")
async def list_products(tenant_id: int, db = Depends(get_db)):
    result = await db.execute("SELECT * FROM products WHERE tenant_id = :tid", {"tid": tenant_id})
    return [dict(row) for row in result.fetchall()]

@router.post("/import_products/")
async def import_products(tenant_id: int, file: UploadFile = File(...), db = Depends(get_db)):
    df = pd.read_csv(file.file)
    for _, row in df.iterrows():
        prod = Product(
            tenant_id=tenant_id,
            name=row['name'],
            description=row.get('description', ''),
            color=row.get('color', ''),
            price=row.get('price', 0.0),
            is_rental=row.get('is_rental', False),
            available_stock=row.get('available_stock', 0),
            image_url=row.get('image_url', ''),
        )
        db.add(prod)
    await db.commit()
    return {"status": "imported"}

@router.get("/orders/")
async def list_orders(tenant_id: int, db = Depends(get_db)):
    result = await db.execute("SELECT * FROM orders WHERE tenant_id = :tid", {"tid": tenant_id})
    return [dict(row) for row in result.fetchall()]

@router.get("/analytics/orders_per_day")
async def orders_per_day(tenant_id: int, db = Depends(get_db)):
    result = await db.execute(
        "SELECT DATE(created_at) as date, COUNT(*) FROM orders WHERE tenant_id = :tid GROUP BY date ORDER BY date DESC",
        {"tid": tenant_id}
    )
    return [{"date": row[0], "orders": row[1]} for row in result.fetchall()]

@router.post("/catalog/upload/")
async def upload_catalog(
    tenant_id: int,                         # from query param or auth
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    # Check file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files supported")

    # Read file into pandas DataFrame
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    required_cols = ["name_en", "price"]  # Expand as needed
    if not all(col in df.columns for col in required_cols):
        raise HTTPException(
            status_code=400,
            detail=f"CSV must include: {', '.join(required_cols)}"
        )

    # Insert products for this tenant
    products = []
    for row in df.to_dict(orient="records"):
        prod = Product(
            tenant_id=tenant_id,
            name_en=row["name_en"],
            name_hi=row.get("name_hi"),
            name_gu=row.get("name_gu"),
            description_en=row.get("description_en"),
            description_hi=row.get("description_hi"),
            description_gu=row.get("description_gu"),
            color=row.get("color"),
            type=row.get("type"),
            price=float(row["price"]),
            is_rental=bool(row.get("is_rental", False)),
            available_stock=int(row.get("available_stock", 0)),
            image_url=row.get("image_url"),
            metadata=row.get("metadata"),
        )
        db.add(prod)
        products.append(prod)
    await db.commit()

    # TODO: Optionally trigger product embedding job here (Pinecone)
    return {"inserted": len(products), "message": "Catalog uploaded!"}