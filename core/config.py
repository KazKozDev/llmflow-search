"""
Configuration models with Pydantic validation.
Provides type-safe, validated configuration loading.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Optional
import json
from pathlib import Path


class LLMConfig(BaseModel):
    """LLM service configuration."""
    provider: str = Field(..., description="LLM provider (ollama, openai, gemini, etc.)")
    model: str = Field(..., description="Model name")
    temperature: float = Field(0.2, ge=0, le=2, description="Sampling temperature")
    max_tokens: int = Field(4096, gt=0, le=128000, description="Maximum tokens")


class CacheConfig(BaseModel):
    """Cache configuration."""
    provider: str = Field("sqlite", description="Cache provider")
    sqlite_path: str = Field("./data/cache.db", description="SQLite database path")
    ttl_seconds: int = Field(86400, gt=0, description="Time-to-live in seconds")
    compress: bool = Field(True, description="Enable compression")


class RateLimitConfig(BaseModel):
    """Rate limit configuration for a tool."""
    requests_per_minute: int = Field(30, gt=0, le=1000)


class SearchConfig(BaseModel):
    """Search behavior configuration."""
    max_results: int = Field(5, gt=0, le=20)
    parse_top_results: int = Field(3, gt=0, le=10)
    use_selenium: bool = Field(True)
    use_cache: bool = Field(True)


class MemoryConfig(BaseModel):
    """Memory module configuration."""
    path: str = Field("./memory", description="Memory storage path")
    max_items: int = Field(100, gt=0, le=10000)


class ReportConfig(BaseModel):
    """Report generation configuration."""
    output_file: str = Field("report.md")
    include_sources: bool = Field(True)
    max_source_length: int = Field(3000, gt=0)


class IntentAnalyzerConfig(BaseModel):
    """Intent analyzer configuration."""
    enabled: bool = Field(True)
    cache_results: bool = Field(True)


class WebServerConfig(BaseModel):
    """Web server configuration."""
    host: str = Field("0.0.0.0")
    port: int = Field(8000, gt=0, lt=65536)
    reload: bool = Field(False)


class AppConfig(BaseModel):
    """Complete application configuration."""
    llm: LLMConfig
    cache: CacheConfig
    rate_limits: Dict[str, RateLimitConfig] = Field(default_factory=dict)
    search: SearchConfig
    memory: MemoryConfig
    report: ReportConfig
    intent_analyzer: IntentAnalyzerConfig
    web_server: Optional[WebServerConfig] = None

    @validator('rate_limits', pre=True)
    def parse_rate_limits(cls, v):
        """Convert dict of dicts to dict of RateLimitConfig."""
        if isinstance(v, dict):
            return {
                key: val if isinstance(val, RateLimitConfig) else RateLimitConfig(**val)
                for key, val in v.items()
            }
        return v

    @classmethod
    def from_file(cls, config_path: str = "config.json") -> "AppConfig":
        """
        Load configuration from JSON file with validation.
        
        Args:
            config_path: Path to config.json file
            
        Returns:
            Validated AppConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config is invalid
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(**data)

    def model_dump(self) -> dict:
        """Convert config to dictionary (Pydantic v2)."""
        return super().model_dump()
    
    # Alias for backward compatibility
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return self.model_dump()


# Default configuration for fallback
DEFAULT_CONFIG = AppConfig(
    llm=LLMConfig(provider="ollama", model="qwen3:14b"),
    cache=CacheConfig(),
    search=SearchConfig(),
    memory=MemoryConfig(),
    report=ReportConfig(),
    intent_analyzer=IntentAnalyzerConfig()
)


def load_config(config_path: str = "config.json") -> AppConfig:
    """
    Load and validate configuration.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Validated AppConfig
    """
    try:
        return AppConfig.from_file(config_path)
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found, using defaults")
        return DEFAULT_CONFIG
    except Exception as e:
        print(f"Error loading config: {e}")
        raise
