
import uvicorn

from app.api.routes import app
from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)