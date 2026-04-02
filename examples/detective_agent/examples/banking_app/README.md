# SentinelBank AI: Multi-Agent Banking System

## 💡 What is SentinelBank AI?

SentinelBank AI is an intelligent banking assistant that thinks like a team of financial experts working together. Imagine having a personal banker, fraud detective, investment advisor, and compliance officer all collaborating in real-time to handle your financial needs.

**Simply put:** Ask the system anything in plain English - "Transfer $500 to my savings," "Is this investment sustainable?", or "Show me my spending patterns" - and specialized AI agents automatically coordinate to complete your request safely and efficiently.

**Key Benefits:**

- 🗣️ **Talk naturally** - No complex banking forms or menus
- 🛡️ **Ultra-secure** - AI constantly monitors for fraud and ensures compliance
- 🚀 **Smart insights** - Get personalized recommendations based on your financial behavior
- 🔗 **Connected** - Works with traditional banks and investment platforms

---

## 🏗 Technical Overview

SentinelBank AI is a next-generation, autonomous banking application built on a multi-agent architecture. It leverages the Model Context Protocol (MCP) to provide agents with real-time access to financial data and utilizes Agent-to-Agent (A2A) communication protocols to coordinate complex financial workflows securely and efficiently.

The system is designed to be accessed via a robust Command Line Interface (CLI) and a RESTful API.

## 🏗 Architecture

The application follows a decentralized multi-agent design where specific agents handle specialized domains:

1. **Orchestrator Agent**: The primary entry point with natural language processing. It parses user intent, routes tasks to specialized agents, and aggregates responses.
2. **Ledger Agent**: Manages account balances, transaction history, and database interactions via MCP-connected tools.
3. **Security & Fraud Agent**: AI-powered real-time fraud detection with ML models for risk scoring and biometric identity verification.
4. **Notification Agent**: Handles asynchronous alerts, receipt generation, and multi-channel communication.
5. **Market Agent**: Real-time market data integration, investment tracking, and robo-advisory services.

### Communication Flow (A2A)

Agents communicate using a standardized A2A protocol with intelligent routing and parallel processing capabilities. Examples of workflows:

**Transfer Request:**
Orchestrator → Security Agent (Risk Analysis) → Ledger Agent (Execute) → Notification Agent (Confirm)

**Investment Advisory:**
Orchestrator → Market Agent (Data Retrieval) → Ledger Agent (Execute if Approved)

**Fraud Detection:**
Security Agent → Notification Agent (Alert) → Orchestrator (Response)

### System Architecture Overview

```mermaid
graph TB
    subgraph "Presentation Layer"
        UserInterface[User Interfaces<br/>CLI & REST API]
    end

    subgraph "Multi-Agent System"
        Orchestrator[Orchestrator Agent<br/>Intent Parsing & Routing]

        subgraph "Specialized Agents"
            Ledger[Ledger Agent<br/>Account & Transaction Management]
            Security[Security Agent<br/>Fraud Detection & Risk Analysis]
            Market[Market Agent<br/>Market Data & Investment Advisory]
            Notification[Notification Agent<br/>Alerts & Communications]
        end

        AgentCommunication[A2A Communication Protocol]
    end

    subgraph "Service Layer"
        MCPServers[MCP Servers<br/>Banking Core, Fraud Detection, Market Data]
    end

    subgraph "Data Layer"
        DataSources[Data Storage<br/>PostgreSQL, Redis, Vector DB]
    end

    subgraph "External Layer"
        ExternalSystems[External Systems<br/>Banks, Markets, AI Providers]
    end

    subgraph "Security Layer"
        SecuritySystems[Security & Compliance<br/>Auth, Encryption, Audit]
    end

    Orchestrator --> AgentCommunication
    AgentCommunication --> Ledger
    AgentCommunication --> Security
    AgentCommunication --> Market
    AgentCommunication --> Notification

    Ledger --> MCPServers
    Security --> MCPServers
    Market --> MCPServers

    MCPServers --> DataSources
    MCPServers --> ExternalSystems

    %% Security flows (dotted lines)
    UserInterface -.-> SecuritySystems
    AgentCommunication -.-> SecuritySystems
    MCPServers -.-> SecuritySystems
    DataSources -.-> SecuritySystems

    %% Styling
    classDef presentation fill:#e3f2fd
    classDef agents fill:#f3e5f5
    classDef service fill:#e8f5e8
    classDef data fill:#fff3e0
    classDef external fill:#fce4ec
    classDef security fill:#f1f8e9

    class UserInterface presentation
    class Orchestrator,Ledger,Security,Market,Notification,AgentCommunication agents
    class MCPServers service
    class DataSources data
    class ExternalSystems external
    class SecuritySystems security
```

## 🚀 Features

### Core Banking

- **MCP Integration**: Seamlessly connects AI models to secure banking databases and external financial APIs
- **Multi-Agent Coordination**: Complex tasks are broken down and handled by specialized autonomous units
- **Dual-Interface**: Full functionality available via CLI for developers and API for integration
- **Security First**: Every A2A interaction is signed and validated by the Security Agent

### AI-Powered Intelligence

- **Real-time Fraud Detection**: Advanced ML models for transaction anomaly detection
- **Natural Language Processing**: Text-based banking interactions with context understanding

### Modern Banking Services

- **Real-time Market Data**: Live financial market integration with investment tracking
- **Robo-Advisory**: Automated investment portfolio management with risk optimization

### Enhanced Security & Compliance

- **Real-time AML/KYC monitoring**: Basic compliance checks with transaction screening
- **Zero-Trust Architecture**: End-to-end encryption with continuous security validation
- **Audit Trail**: Secure transaction logging and verification

## 🛠 Tech Stack

### Core Technologies

- **Language**: Python 3.11+ with async/await support
- **Framework**: FastAPI (for API) with WebSocket support
- **AI Protocol**: Model Context Protocol (MCP) v2.0
- **Agent Framework**: LangGraph / CrewAI (Enhanced A2A Communication)
- **Database**: PostgreSQL with TimescaleDB for time-series data

### AI & Machine Learning

- **ML Framework**: PyTorch / TensorFlow for fraud detection models
- **LLM Integration**: OpenAI GPT-4, Anthropic Claude, local LLama models
- **Vector Database**: Pinecone / Chroma for semantic search and embeddings
- **MLOps**: MLflow for model versioning and deployment

### Security & Authentication

- **Authentication**: OAuth 2.0 for secure access
- **Encryption**: AES-256, RSA-4096 for data protection
- **Zero-Trust**: HashiCorp Vault for secrets management

### Data & Integration

- **Message Queue**: Redis Streams for real-time event processing
- **API Gateway**: Kong for rate limiting and API management
- **Monitoring**: Prometheus + Grafana for observability
- **Financial Data**: Alpha Vantage, Yahoo Finance for market data

### Infrastructure

- **Containerization**: Docker with multi-stage builds
- **Orchestration**: Kubernetes for production deployment
- **CI/CD**: GitHub Actions with automated testing
- **Cloud**: Multi-cloud support (AWS, GCP, Azure)

## 📂 Project Structure

```
sentinel-bank/
├── agents/
│   ├── orchestrator.py        # Main entry agent with NLP
│   ├── ledger.py             # Account & transaction management
│   ├── security.py           # AI-powered fraud detection
│   ├── notification.py       # Multi-channel alerts
│   └── market.py            # Real-time market data & robo-advisory
├── mcp/
│   ├── servers/
│   │   ├── banking_core.py    # Core banking operations MCP
│   │   ├── fraud_detection.py # ML-powered fraud detection MCP
│   │   └── market_data.py     # Financial market data MCP
│   ├── tools.py              # MCP tool definitions
│   └── client.py             # MCP client implementation
├── ai/
│   ├── models/
│   │   ├── fraud_detector.py  # ML fraud detection models
│   │   ├── risk_scorer.py     # Credit & investment risk models
│   │   ├── nlp_processor.py   # Natural language processing
│   │   └── recommendation.py  # Personalized recommendation engine
│   ├── vectordb/             # Vector database for embeddings
│   └── training/             # Model training scripts
├── api/
│   ├── v1/
│   │   ├── auth.py           # Authentication endpoints
│   │   ├── banking.py        # Core banking operations
│   │   ├── analytics.py      # Financial insights APIs
│   │   └── investments.py    # Investment & advisory APIs
│   ├── websockets.py         # Real-time WebSocket connections
│   └── main.py              # FastAPI application entry
├── cli/
│   ├── interface.py         # Enhanced CLI interface
│   ├── admin.py            # Administrative commands
│   └── developer.py        # Developer tools & debugging
├── security/
│   ├── auth/               # Multi-factor authentication
│   ├── encryption.py      # Encryption utilities
│   ├── vault.py          # Secrets management
│   └── audit.py          # Security audit logging
├── config/
│   ├── agents.yaml        # Agent configurations
│   ├── mcp_servers.yaml   # MCP server configurations
│   ├── ml_models.yaml     # ML model configurations
│   └── security.yaml      # Security configurations
├── tests/
│   ├── integration/       # Integration tests
│   ├── agents/           # Agent-specific tests
│   └── mcp/             # MCP server tests
├── docs/
│   ├── api/             # API documentation
│   ├── agents/          # Agent documentation
│   └── deployment/      # Deployment guides
└── main.py              # Unified application entry point
```

   

## ⚙️ Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis 7+
- Docker & Docker Compose
- UV (recommended) or pip for package management

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-repo/sentinel-bank.git
   cd sentinel-bank
   ```

2. **Install UV (recommended for 10-100x faster installs):**

   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Or via Homebrew
   brew install uv
   ```

3. **Install dependencies:**

   **Option A: Using UV (Recommended)**
   ```bash
   # One command to set up everything
   make setup

   # Or manually
   uv sync
   ```

   **Option B: Using pip (Traditional)**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file:

   ```env
   # Core Configuration
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   DATABASE_URL=postgresql://user:pass@localhost/sentinel_bank
   REDIS_URL=redis://localhost:6379

   # MCP Server Configuration
   MCP_BANKING_SERVER_URL=http://localhost:8001
   MCP_FRAUD_SERVER_URL=http://localhost:8002
   MCP_MARKET_SERVER_URL=http://localhost:8003

   # Security Configuration
   JWT_SECRET_KEY=your_jwt_secret
   VAULT_TOKEN=your_vault_token
   ENCRYPTION_KEY=your_encryption_key

   # Financial APIs
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
   ```

5. **Initialize services:**

   ```bash
   # Start infrastructure services
   docker-compose up -d postgres redis

   # Run database migrations
   python -m alembic upgrade head

   # Initialize AI models
   python scripts/init_ml_models.py

   # Start MCP servers
   python scripts/start_mcp_servers.py
   ```

## 🖥 Usage

   

## 🖥 Usage

### Command Line Interface (CLI)

The CLI supports natural language commands for banking operations.

**Basic Banking Operations:**

```bash
# Check balance with natural language
python main.py cli "What's my checking account balance?"

# Traditional command format
python main.py cli --action balance --account-id 12345

# Transfer with fraud detection
python main.py cli --action transfer --amount 500 --to 67890
```

**AI-Powered Analytics:**

```bash
# Investment recommendations
python main.py cli --action recommend-investments --risk-tolerance moderate

# Fraud risk assessment
python main.py cli --action risk-score --transaction-id txn_123
```

**Notification Operations:**

```bash
# Send a simple notification
python -m cli.main notify --user-id user_123 --title "Payment Received" \
  --message "You received $100 from John Doe"

# Send critical notification with multiple channels
python -m cli.main notify --user-id user_123 --title "Security Alert" \
  --message "Unusual activity detected" --priority critical \
  --channels email,sms,push

# Check available notification channels
python -m cli.main agents  # List all agents including notification agent
```

### REST API

Start the API server with all MCP services:

```bash
python main.py serve --with-mcp-servers
```

**Core Banking Endpoints:**

````http
# Natural language banking
POST /api/v1/chat
Content-Type: application/json

{
  "user_id": "user_01",
  "message": "Transfer $100 to account 99887 and notify me when done",
  "context": "main_banking"
}

**Enhanced transfer with fraud detection**
```http
POST /api/v1/transfers
{
  "from_account": "acc_123",
  "to_account": "acc_456",
  "amount": 1000.00,
  "currency": "USD",
  "fraud_check": true
}
````

````

**AI Analytics Endpoints:**

```http
# Get market insights
GET /api/v1/market/insights?user_id=user_01&timeframe=30d

# Investment recommendations
POST /api/v1/investments/recommend
{
  "user_id": "user_01",
  "investment_amount": 5000,
  "risk_tolerance": "moderate"
}

# Real-time fraud scoring
POST /api/v1/security/fraud-score
{
  "transaction": {...},
  "user_context": {...}
}
````

**Notification Endpoints:**

```http
# Send notification to user
POST /api/v1/notifications
{
  "user_id": "user_123",
  "title": "Payment Received",
  "message": "You received $100 from John Doe",
  "priority": "normal",
  "channels": ["email", "in_app"]
}

# Get available notification channels
GET /api/v1/notifications/channels

# Get user notification preferences
GET /api/v1/notifications/preferences/user_123

# Update user notification preferences
PUT /api/v1/notifications/preferences/user_123
{
  "channels": {
    "email": true,
    "sms": false,
    "push": true
  }
}
```

**WebSocket Connections:**

```javascript
// Real-time notifications and market data
const ws = new WebSocket("ws://localhost:8000/ws/user_01");

// Subscribe to real-time events
ws.send(
  JSON.stringify({
    action: "subscribe",
    channels: ["transactions", "fraud_alerts", "market_updates"],
  }),
);
```

## 🔒 Security and Compliance

### Security Architecture

SentinelBank AI implements a security-first approach with multiple layers of protection:

**Authentication & Authorization:**

- OAuth 2.0 authentication with JWT tokens
- Role-based access control (RBAC)
- Session management and timeout controls

**Fraud Detection:**

- Real-time ML-powered transaction analysis
- Anomaly detection using machine learning models
- Behavioral pattern recognition
- Risk scoring with transparent algorithms

**Agent-to-Agent Security:**

- Cryptographic signatures for all A2A communications
- TLS encryption between agents
- Secure message validation and verification

**Data Protection:**

- Field-level encryption for sensitive data (AES-256)
- Secure key management with HashiCorp Vault
- Regular security audits and monitoring
- Secure audit logs for all transactions

### Basic Compliance Features

**Transaction Monitoring:**

- Real-time transaction screening
- Basic AML (Anti-Money Laundering) checks
- Transaction reporting and audit trails
- Suspicious activity detection

## 🔌 MCP Server Architecture

SentinelBank AI leverages a simplified MCP (Model Context Protocol) server architecture to provide essential AI capabilities:

### Core MCP Servers

**Banking Core MCP (`mcp/servers/banking_core.py`)**

- Account management and transaction processing
- Balance inquiries and account history
- Payment processing with real-time settlements
- Multi-currency support with automatic conversion

**Fraud Detection MCP (`mcp/servers/fraud_detection.py`)**

- Real-time transaction risk assessment
- ML-powered anomaly detection
- Behavioral pattern analysis
- Transaction scoring and validation

**Market Data MCP (`mcp/servers/market_data.py`)**

- Real-time stock, forex, and crypto prices
- Portfolio valuation and performance tracking
- Market news sentiment analysis
- Economic indicators and calendar events

### MCP Server Benefits

- **Scalability**: Independent scaling of AI capabilities
- **Modularity**: Focused architecture for core banking services
- **Performance**: Optimized models for specific use cases
- **Security**: Isolated execution environments
- **Reliability**: Simplified architecture for better stability

## 📄 License

Distributed under the MIT License. See LICENSE for more information.