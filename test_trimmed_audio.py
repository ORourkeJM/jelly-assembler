"""
Test script for trimmed audio extraction and upload flow.
"""
import asyncio
import os
import sys
import tempfile
import httpx

# Test configuration
ASSEMBLER_URL = "https://assembler.jellybop.ai"
TEST_PROJECT_ID = "test-trim-flow-001"

# Sample audio URL (using a public test file)
# This is a Creative Commons music sample
TEST_AUDIO_URL = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"


async def test_trimmed_audio_flow():
    """Test the full trimmed audio extraction and upload flow."""
    print("=" * 60)
    print("TRIMMED AUDIO FLOW TEST")
    print("=" * 60)
    
    # Step 1: Download a small portion of test audio
    print("\n[1/4] Downloading test audio...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Just get the headers first to confirm URL works
            resp = await client.head(TEST_AUDIO_URL, follow_redirects=True)
            print(f"  Audio URL accessible: {resp.status_code == 200}")
            
            # Download first 500KB for testing (enough for a short clip)
            headers = {"Range": "bytes=0-512000"}
            resp = await client.get(TEST_AUDIO_URL, headers=headers, follow_redirects=True)
            
            # Save to temp file
            temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_audio.write(resp.content)
            temp_audio.close()
            print(f"  Downloaded: {len(resp.content)} bytes to {temp_audio.name}")
            
    except Exception as e:
        print(f"  ERROR downloading audio: {e}")
        return False
    
    # Step 2: Test upload to jelly-assembler
    print("\n[2/4] Testing upload to jelly-assembler...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(temp_audio.name, "rb") as f:
                files = {"file": (f"trimmed_{TEST_PROJECT_ID[:8]}.mp3", f, "audio/mpeg")}
                data = {"folder": f"projects/{TEST_PROJECT_ID}/audio"}
                
                resp = await client.post(
                    f"{ASSEMBLER_URL}/api/v1/upload",
                    files=files,
                    data=data
                )
                
                if resp.status_code == 200:
                    result = resp.json()
                    uploaded_url = result.get("url")
                    print(f"  Upload SUCCESS!")
                    print(f"  URL: {uploaded_url}")
                else:
                    print(f"  Upload FAILED: {resp.status_code}")
                    print(f"  Response: {resp.text}")
                    return False
                    
    except Exception as e:
        print(f"  ERROR uploading: {e}")
        return False
    finally:
        # Cleanup temp file
        os.unlink(temp_audio.name)
    
    # Step 3: Verify the uploaded file is accessible
    print("\n[3/4] Verifying uploaded file is accessible...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.head(uploaded_url, follow_redirects=True)
            if resp.status_code == 200:
                print(f"  File accessible: YES")
                content_length = resp.headers.get("content-length", "unknown")
                print(f"  Size: {content_length} bytes")
            else:
                print(f"  File accessible: NO (status {resp.status_code})")
                return False
    except Exception as e:
        print(f"  ERROR verifying: {e}")
        return False
    
    # Step 4: Summary
    print("\n[4/4] Summary")
    print("=" * 60)
    print("TRIMMED AUDIO FLOW: SUCCESS")
    print(f"  - Audio downloaded and saved to temp file")
    print(f"  - Uploaded to R2 via jelly-assembler")
    print(f"  - File accessible at public URL")
    print(f"  - URL: {uploaded_url}")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_trimmed_audio_flow())
    sys.exit(0 if success else 1)
