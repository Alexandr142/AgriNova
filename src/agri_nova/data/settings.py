from pydantic_settings import BaseSettings, SettingsConfigDict
from xarray.backends import lru_cache


class DataSettings(BaseSettings):
    """Fields for data fetching such as logins, passwords etc."""

    earthdata_username: str
    earthdata_password: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )


data_settings = DataSettings()
