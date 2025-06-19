# Sup - WhatsApp-like Messaging Application

A high-performance, real-time messaging application built for maximum speed, fault tolerance, and cross-platform support.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │  Mobile Client  │    │ Desktop Client  │
│ (React Native)  │    │ (React Native)  │    │   (Electron)    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │     Load Balancer         │
                    │    (HAProxy/Nginx)        │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │    Elixir Backend         │
                    │  (Plug/Cowboy + OTP)      │
                    │                           │
                    │  ┌─────────────────────┐  │
                    │  │   Auth Service      │  │
                    │  │   Message Service   │  │
                    │  │   Presence Service  │  │
                    │  │   Room Service      │  │
                    │  │   Push Service      │  │
                    │  └─────────────────────┘  │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │      Data Layer           │
                    │                           │
                    │  ┌─────────┐ ┌─────────┐  │
                    │  │PostgreSQL│ │ScyllaDB │  │
                    │  │(metadata) │(messages)│  │
                    │  └─────────┘ └─────────┘  │
                    │                           │
                    │  ┌─────────┐ ┌─────────┐  │
                    │  │  Redis  │ │ Vector  │  │
                    │  │(sessions)│ │   DB    │  │
                    │  └─────────┘ └─────────┘  │
                    └───────────────────────────┘
```

## 🚀 Tech Stack

**Backend:**
- **Elixir/OTP** - Raw OTP with Plug/Cowboy (No Phoenix)
- **PostgreSQL** - User accounts, room metadata, relationships
- **ScyllaDB** - High-throughput message storage
- **Redis** - Sessions, presence cache, real-time state
- **Vector DB** - Semantic search and AI features

**Frontend:**
- **React Native + Expo** - Cross-platform (Web, Mobile, Desktop)
- **TypeScript** - Type safety and modern patterns
- **Electron** - Desktop wrapper
- **WebSockets** - Real-time communication

## 📁 Project Structure

```
sup/
├── backend/                    # Elixir backend
├── frontend/                   # React Native + Expo
├── shared/                     # Shared types/schemas
├── docker/                     # Docker configurations
├── k8s/                       # Kubernetes manifests
└── docs/                      # Documentation
```

## 🔧 Key Features

- **Real-time messaging** with delivery receipts
- **Presence detection** (online/typing indicators)  
- **Group chats** and 1:1 messaging
- **End-to-end encryption** hooks (future)
- **Message search** (basic + semantic)
- **Push notifications**
- **AI-powered smart replies**
- **Cross-platform support**

## 🏃‍♂️ Quick Start

### Prerequisites

- **Elixir 1.15+** with OTP 26+
- **Node.js 18+** with npm
- **PostgreSQL 15+**
- **Redis 7+**
- **ScyllaDB 5.2+** (optional, will auto-setup)

### Option 1: Docker Setup (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd Sup

# Start all services with Docker
docker-compose up -d

# The app will be available at:
# Backend API: http://localhost:4000
# Frontend Web: http://localhost:19006
```

### Option 2: Manual Setup

#### Backend Setup

```bash
cd backend

# Copy environment file
cp .env.example .env

# Install dependencies
mix deps.get

# Setup database
mix ecto.setup

# Start the server
mix run --no-halt
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start

# For mobile development
npm run android  # Android
npm run ios      # iOS

# For desktop (Electron)
npm run electron
```

### Production Deployment

#### Using Docker

```bash
# Build production images
docker-compose -f docker-compose.prod.yml up -d
```

#### Manual Production Setup

```bash
# Backend
cd backend
MIX_ENV=prod mix release
_build/prod/rel/sup/bin/sup start

# Frontend
cd frontend
npm run build:web  # Web build
npm run build:android  # Android APK
npm run build:ios  # iOS build
```

## 🧪 Testing

### Backend Tests

```bash
cd backend
mix test
```

### Frontend Tests

```bash
cd frontend
npm test
```

### End-to-End Tests

```bash
# Start services
docker-compose up -d

# Run E2E tests
npm run test:e2e
```

## 📖 Documentation

- [System Architecture](docs/architecture.md)
- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Development Setup](docs/development.md)
- [WebSocket Protocol](docs/websocket.md)
- [Database Schema](docs/database.md)
