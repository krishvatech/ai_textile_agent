import os
import re
import html as ihtml
from decimal import Decimal, InvalidOperation
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv
from textwrap import fill

load_dotenv()

SHOP_NAME = os.getenv("SHOP_NAME")                 # e.g. i9egty-pf
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")           # Admin API access token
APP_NAME = os.getenv("SHOPIFY_APP_NAME", "MyApp")  # Optional: your app name for User-Agent

def get_product_by_handle(shop_name: str, access_token: str, handle: str) -> dict | None:
    api_url = f"https://{shop_name}.myshopify.com/admin/api/2024-07/products.json"
    headers = {
        "X-Shopify-Access-Token": access_token,
        "Content-Type": "application/json",
        "User-Agent": f"{APP_NAME} - ProductFetcher",
    }
    params = {"handle": handle}
    resp = requests.get(api_url, headers=headers, params=params, timeout=20)
    if resp.ok:
        products = resp.json().get("products", [])
        return products[0] if products else None
    else:
        print("Shopify API error:", resp.status_code, resp.text)
    return None

def extract_product_fields(product: dict) -> dict:
    name = (product or {}).get("title", "").strip()

    body_html = (product or {}).get("body_html") or ""
    desc_text = re.sub(r"<[^>]+>", "", body_html)
    description = ihtml.unescape(desc_text).strip() or None

    price_decimal = None
    for v in (product or {}).get("variants", []):
        try:
            p = Decimal(v.get("price", "0"))
            price_decimal = p if price_decimal is None or p < price_decimal else price_decimal
        except (InvalidOperation, TypeError):
            continue
    price = f"{price_decimal:.2f}" if price_decimal is not None else None

    fabric = color = None
    option_pos_to_name = {
        opt.get("position"): (opt.get("name") or "").strip()
        for opt in (product or {}).get("options", [])
        if isinstance(opt, dict)
    }
    first_variant = (product or {}).get("variants", [{}])[0] if (product or {}).get("variants") else {}
    values_by_name_lower = {}
    for pos in (1, 2, 3):
        name_here = option_pos_to_name.get(pos)
        val_here = first_variant.get(f"option{pos}")
        if name_here and val_here:
            values_by_name_lower[name_here.lower()] = val_here

    fabric = values_by_name_lower.get("fabric")
    color = values_by_name_lower.get("color")

    if not fabric:
        for k, v in values_by_name_lower.items():
            if "fabric" in k.replace(" ", "").lower():
                fabric = v
                break

    if not color:
        for k, v in values_by_name_lower.items():
            if "color" in k.replace(" ", "").lower():
                color = v
                break

    return {
        "name": name or None,
        "price": price,
        "fabric": fabric,
        "color": color,
        "description": description,
    }

