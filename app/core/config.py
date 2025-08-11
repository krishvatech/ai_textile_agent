import os

def str_to_bool(v: str | None) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y", "on"} if v is not None else False

class Settings:
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_TLS: bool = str_to_bool(os.getenv("REDIS_TLS", "false"))

    @property
    def redis_dsn(self) -> str:
        """
        Build final DSN:
        - honor REDIS_URL if provided
        - switch to rediss:// if REDIS_TLS=true
        - ensure DB suffix '/<db>'
        """
        url = self.REDIS_URL.strip()
        # swap scheme if TLS requested
        if self.REDIS_TLS and url.startswith("redis://"):
            url = "rediss://" + url[len("redis://"):]
        # append db if missing
        if "/" not in url.split("://", 1)[1]:
            url = url.rstrip("/") + f"/{self.REDIS_DB}"
        return url

settings = Settings()
