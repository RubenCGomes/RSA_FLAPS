import os
from pathlib import Path

import uvicorn
from interface.interface import mflinterface_app
from separate import router as separate_router

# Ensure predict() output lands in the logs volume, not a random /runs dir
from clientML import APP
APP.run_path = Path(os.environ.get("LOG_DIR", "/app/logs")) / "separations"
APP.run_path.mkdir(parents=True, exist_ok=True)

mflinterface_app.include_router(separate_router)

if __name__ == "__main__":
    uvicorn.run(
        "main:mflinterface_app",
        host="0.0.0.0",
        port=5000,
        reload=False,
    )