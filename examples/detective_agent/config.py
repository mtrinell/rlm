"""Configuration management using Pydantic Settings."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(Path(__file__).resolve().parent / ".env")


class Settings(BaseSettings):
    """Application settings with validation."""

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
        default=200_000,
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

    # ── Input: Dataset & Spec ─────────────────────────────────────────────────
    dataset_path: str = Field(
        default="examples/banking_app/otel-traces.jsonl",
        description=(
            "Path to the dataset to analyse. Can be a single file (JSON, JSONL, YAML, CSV, "
            "log, …), a directory, or an archive (.tar.gz, .zip, …). "
            "Relative paths are resolved from the detective_agent directory."
        ),
    )
    spec_path: str = Field(
        description=(
            "Path to a specification describing the expected behaviour of the system under analysis. "
            "Can be a single file (README, API spec, schema, OpenAPI YAML, …), "
            "a directory (e.g. a full project source tree), or an archive (.tar.gz, .zip, …). "
            "The RLM will explore its contents to infer what the application is meant to do. "
            "Relative paths are resolved from the detective_agent directory."
        ),
    )

    @field_validator("spec_path")
    @classmethod
    def spec_path_required(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "SPEC_PATH is required — provide a path to a specification file, directory, or archive."
            )
        return v
    investigation_focus: str = Field(
        default="",
        description=(
            "Optional hint for the investigation (e.g. 'focus on the security agent'). "
            "Leave empty for a full scan."
        ),
    )
    user_prompt: str = Field(
        default="",
        description=(
            "The end-user's original question or investigation directive. "
            "When provided, the analyst also answers this question directly "
            "and flags any gaps as findings."
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
    def resolved_dataset_path(self) -> Path:
        """Absolute path to the dataset (file, directory, or archive)."""
        p = Path(self.dataset_path)
        return p if p.is_absolute() else (Path(__file__).parent / p).resolve()

    @property
    def resolved_spec_path(self) -> Path:
        """Absolute path to the spec/documentation file or directory."""
        p = Path(self.spec_path)
        return p if p.is_absolute() else (Path(__file__).parent / p).resolve()


# Global settings instance — imported by main.py and tests
settings = Settings()
