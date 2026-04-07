"""Configuration management using Pydantic Settings."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""

    # ── Azure OpenAI ──────────────────────────────────────────────────────────
    azure_openai_api_key: str = Field(..., description="Azure OpenAI API Key")
    azure_openai_endpoint: str = Field(..., description="Azure OpenAI Endpoint URL")
    azure_openai_api_version: str = Field(
        default="2024-02-15-preview",
        description="Azure OpenAI API Version",
    )
    azure_openai_model: str = Field(
        default="gpt-4o",
        description="Model deployment name (e.g., gpt-4o, gpt-4o-mini)",
    )

    # ── RLM Execution ─────────────────────────────────────────────────────────
    max_iterations: int = Field(
        default=20,
        description="Maximum top-level LLM iterations",
        ge=1,
        le=100,
    )
    max_depth: int = Field(
        default=1,
        description="Maximum recursive sub-call depth (1 = single model)",
        ge=1,
        le=5,
    )

    # ── Context Management ────────────────────────────────────────────────────
    context_token_limit: int = Field(
        default=100_000,
        description=(
            "Token budget ceiling for progressive context truncation. "
            "Truncation tiers activate at 50/70/85% of this value."
        ),
        ge=10_000,
    )
    phase_injection_enabled: bool = Field(
        default=True,
        description="Inject phase-boundary reminder messages when context pressure increases.",
    )
    helpers_injection_enabled: bool = Field(
        default=True,
        description="Inject filesystem helper functions into the REPL namespace.",
    )

    # ── Input: App Under Test ─────────────────────────────────────────────────
    app_readme_path: str = Field(
        default="examples/banking_app/README.md",
        description=(
            "Path to the README / documentation file of the application under test "
            "(relative to the detective_agent project root or absolute)."
        ),
    )
    traces_path: str = Field(
        default="examples/banking_app/otel-traces.jsonl",
        description=(
            "Path to the OpenTelemetry JSONL traces file produced by the application under test "
            "(relative to the detective_agent project root or absolute)."
        ),
    )
    investigation_focus: str = Field(
        default="",
        description=(
            "Optional hint for the investigation (e.g. 'focus on the security agent'). "
            "Leave empty for a full behavioral compliance scan."
        ),
    )
    user_prompt: str = Field(
        default="",
        description=(
            "The end-user's original request to the application under test. "
            "When provided, the detector also checks whether the app fulfilled this request "
            "and flags deviations as DERAILMENT_USER findings."
        ),
    )

    # ── Logging / Output ─────────────────────────────────────────────────────
    log_dir: str = Field(
        default="results",
        description="Directory for JSONL iteration logs and result files (relative to project root).",
    )
    verbose: bool = Field(
        default=True,
        description="Enable rich console output during RLM execution.",
    )

    # Resolve .env relative to project root (2 levels up from src/rlm/config.py)
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent / ".env"),
        extra="ignore",
        case_sensitive=False,
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("azure_openai_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or v == "your-api-key-here":
            raise ValueError(
                "AZURE_OPENAI_API_KEY is not set or is using a placeholder value. "
                "Please update your .env file with a valid API key.",
            )
        return v

    @field_validator("azure_openai_endpoint")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        if not v:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required")
        if not v.startswith("https://"):
            raise ValueError(f"AZURE_OPENAI_ENDPOINT must start with https://, got: {v}")
        if not v.endswith("/"):
            v = v + "/"
        return v

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def project_root(self) -> Path:
        """Absolute path to the project root (parent of src/)."""
        return Path(__file__).parent.parent

    @property
    def resolved_log_dir(self) -> Path:
        """Absolute path to the log/results directory."""
        p = Path(self.log_dir)
        return p if p.is_absolute() else self.project_root / p

    @property
    def resolved_app_readme_path(self) -> Path:
        """Absolute path to the app README / documentation file."""
        p = Path(self.app_readme_path)
        return p if p.is_absolute() else (Path(__file__).parent / p).resolve()

    @property
    def resolved_traces_path(self) -> Path:
        """Absolute path to the OTEL JSONL traces file."""
        p = Path(self.traces_path)
        return p if p.is_absolute() else (Path(__file__).parent / p).resolve()

    @property
    def backend_kwargs(self) -> dict:
        """Convenience property: Azure OpenAI client kwargs dict."""
        return {
            "api_key": self.azure_openai_api_key,
            "azure_endpoint": self.azure_openai_endpoint,
            "api_version": self.azure_openai_api_version,
            "model_name": self.azure_openai_model,
        }


# Global settings instance — imported by main.py and tests
settings = Settings()
