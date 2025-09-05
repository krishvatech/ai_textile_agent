#!/usr/bin/env python3
"""
Local Virtual Try-On Test Script

This script allows you to test the virtual try-on functionality locally
by providing person and garment images from your computer.

Usage:
1. Place person image as 'person.jpg' in the same directory
2. Place garment image as 'garment.jpg' in the same directory  
3. Run: python local_vto_test.py
4. Output will be saved as 'result_vto.png'
"""

import asyncio
import os
import sys
from pathlib import Path

# Add your project root to Python path if needed
# sys.path.append('/path/to/your/project')

try:
    from app.core.virtual_try_on import generate_vto_image, VTOConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from your project root directory")
    sys.exit(1)

async def test_virtual_try_on():
    """Main function to test VTO locally."""
    
    # Check if required images exist
    person_path = "person.jpg"
    garment_path = "garment.jpg"
    
    if not os.path.exists(person_path):
        print(f"❌ Person image not found: {person_path}")
        print("Please add a person image named 'person.jpg' in this directory")
        return
        
    if not os.path.exists(garment_path):
        print(f"❌ Garment image not found: {garment_path}")
        print("Please add a garment image named 'garment.jpg' in this directory")
        return

    print("📸 Loading images...")
    
    # Load person image
    try:
        with open(person_path, "rb") as f:
            person_bytes = f.read()
        print(f"✅ Loaded person image: {len(person_bytes)} bytes")
    except Exception as e:
        print(f"❌ Error loading person image: {e}")
        return

    # Load garment image  
    try:
        with open(garment_path, "rb") as f:
            garment_bytes = f.read()
        print(f"✅ Loaded garment image: {len(garment_bytes)} bytes")
    except Exception as e:
        print(f"❌ Error loading garment image: {e}")
        return

    # Configure VTO settings
    cfg = VTOConfig(
        base_steps=60,          # Higher = better quality, slower
        seed=12345,             # For reproducible results
        add_watermark=False,    # Set to True if you want Google's watermark
        model="virtual-try-on-preview-08-04"  # Google's VTO model
    )
    
    print("🔄 Running virtual try-on...")
    print(f"   Model: {cfg.model}")
    print(f"   Steps: {cfg.base_steps}")
    print(f"   Seed: {cfg.seed}")

    try:
        # Run the VTO process
        result_bytes = await generate_vto_image(
            person_bytes=person_bytes,
            garment_bytes=garment_bytes, 
            cfg=cfg
        )
        
        print(f"✅ VTO completed: {len(result_bytes)} bytes generated")
        
        # Save result
        output_path = "result_vto.png"
        with open(output_path, "wb") as out_file:
            out_file.write(result_bytes)
            
        print(f"💾 Result saved: {output_path}")
        print(f"📁 Full path: {Path(output_path).absolute()}")
        
    except Exception as e:
        print(f"❌ VTO failed: {e}")
        print("\nCommon issues:")
        print("- Check your GOOGLE_API_KEY environment variable")
        print("- Ensure person image shows a clear front-facing person")
        print("- Ensure garment image is well-cropped")
        return

def check_environment():
    """Check if required environment variables are set."""
    
    required_vars = ["GOOGLE_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print("❌ Missing required environment variables:")
        for var in missing:
            print(f"   - {var}")
        print("\nPlease set them in your .env file or environment")
        return False
        
    print("✅ Environment variables configured")
    return True

def main():
    """Entry point."""
    print("🎭 Virtual Try-On Local Test")
    print("=" * 40)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Check current directory
    cwd = Path.cwd()
    print(f"📂 Working directory: {cwd}")
    
    # Run the test
    asyncio.run(test_virtual_try_on())
    
    print("\n✨ Test complete!")

if __name__ == "__main__":
    main()
