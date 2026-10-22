"""
HemoVision AI — Application Entry Point
Usage:
    python main.py
    python main.py --no-browser
"""
import os
import sys
import webbrowser
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    url = f"http://localhost:{port}"

    print("\n" + "=" * 68)
    print("  🩸 HemoVision AI: IDA & Beta Thalassemia Trait Decision Support")
    print(f"  🚀 Web Cockpit:        {url}")
    print(f"  🌐 Network Interface:  http://{host}:{port}")
    print(f"  📖 Interactive API:    http://localhost:{port}/docs")
    print("=" * 68 + "\n")

    # Attempt to open the default web browser on local desktop environments
    if "--no-browser" not in sys.argv and not os.environ.get("HEADLESS"):
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run("server:app", host=host, port=port, reload=False)
