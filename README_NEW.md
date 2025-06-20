# Sup - Real-time Messaging Platform

Sup is a modern, feature-rich real-time messaging platform built with Elixir (Phoenix) and React Native. It offers Discord-like functionality with comprehensive user profiles, friend systems, voice/video calls, and AI-powered autocomplete.

## ✨ Features

### 🔐 Authentication & Security
- **Secure Registration/Login** with JWT authentication
- **Multi-step Registration** with profile setup and avatar selection
- **Two-Factor Authentication** (2FA) support
- **Email and Phone Verification**
- **Password Security** with bcrypt hashing

### 👤 User Profiles & Customization
- **Custom Display Names** and usernames
- **Profile Pictures** and banner images
- **Personal Bios** and status messages
- **Activity Status** (Online, Away, Busy, Invisible)
- **Theme Customization** with accent colors
- **Friend Codes** for easy user discovery

### 👥 Social Features
- **Friend System** with friend requests and management
- **User Search** and discovery
- **Block/Unblock** functionality
- **Online Status** and presence indicators
- **Privacy Controls** for profile visibility

### 💬 Messaging
- **Real-time Messaging** with WebSocket connections
- **Direct Messages** and group chats
- **Message History** and search
- **Typing Indicators** and read receipts
- **Rich Text Support** with emoji
- **File Sharing** capabilities

### 📞 Voice & Video Calls
- **WebRTC-powered** voice and video calls
- **Call Management** (accept, reject, end)
- **Audio Controls** (mute, volume, noise suppression)
- **Video Controls** (camera toggle, screen sharing)
- **Call History** and duration tracking

### 🤖 AI-Powered Autocomplete
- **Smart Text Completion** using machine learning
- **Context-Aware Suggestions** based on conversation history
- **Multiple Model Support** (embedder, generator, ranker)
- **Optimized Performance** with trie data structures
- **Microservice Architecture** for scalability

### ⚙️ Settings & Preferences
- **Comprehensive Settings Panel** with modern UI
- **Notification Preferences** (messages, mentions, calls)
- **Privacy Settings** (online status, profile visibility)
- **Call Settings** (camera/mic defaults, quality)
- **Theme Selection** (light, dark, system)
- **Data Management** (storage, export, cache)

### 🔔 Notifications
- **Push Notifications** for mobile and web
- **Sound and Vibration** controls
- **Smart Notification** filtering
- **Do Not Disturb** modes

## 🏗️ Architecture

### Backend (Elixir/Phoenix)
- **Phoenix Framework** for web APIs and WebSocket handling
- **Ecto** for database management with PostgreSQL
- **Guardian** for JWT authentication
- **GenServer** for real-time features
- **ScyllaDB** for high-performance message storage
- **Redis** for caching and session management

### Frontend (React Native/Expo)
- **React Native** for cross-platform mobile apps
- **Expo** for rapid development and deployment
- **TypeScript** for type safety
- **Zustand** for state management
- **React Native Paper** for Material Design UI
- **WebRTC** for real-time communication

### Autocomplete Service (Python)
- **FastAPI** for high-performance API
- **PyTorch** for machine learning models
- **Sentence Transformers** for text embeddings
- **Trie Data Structures** for efficient search
- **Docker** containerization

### Infrastructure
- **Docker & Docker Compose** for containerization
- **Kubernetes** for orchestration and scaling
- **PostgreSQL** for relational data
- **ScyllaDB** for time-series message data
- **Redis** for caching and real-time features
- **NGINX** for load balancing

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for development)
- Elixir 1.14+ and Erlang 25+ (for development)
- Python 3.9+ (for autocomplete service development)

### Option 1: Docker Compose (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/sup.git
cd sup

# Deploy with Docker Compose
./deploy.sh docker deploy

# Access the application
# Frontend: http://localhost:19006
# Backend API: http://localhost:4000
# Autocomplete Service: http://localhost:8000
```

### Option 2: Kubernetes
```bash
# Ensure kubectl is configured
kubectl cluster-info

# Deploy to Kubernetes
./deploy.sh kubernetes deploy

# Check deployment status
kubectl get pods -n sup
kubectl get services -n sup
```

### Option 3: Development Environment
```bash
# Setup development environment
./deploy.sh development deploy

# Start services individually
cd backend && mix phx.server &
cd frontend && npm start &
cd autocomplete && ./activate.sh && python api_server.py &
```

## 📱 Mobile App Development

### React Native Setup
```bash
cd frontend

# Install dependencies
npm install

# iOS (requires Xcode)
npx expo run:ios

# Android (requires Android Studio)
npx expo run:android

# Web
npx expo start --web
```

### Building for Production
```bash
# Build for production
npx expo build:android
npx expo build:ios

# Or use EAS Build
npx eas build --platform all
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/sup_prod
REDIS_URL=redis://localhost:6379
SCYLLA_NODES=localhost:9042

# Backend
SECRET_KEY_BASE=your-secret-key-base
GUARDIAN_SECRET_KEY=your-guardian-secret
MIX_ENV=prod

# Frontend
REACT_APP_API_URL=http://localhost:4000
REACT_APP_WS_URL=ws://localhost:4000

# Autocomplete
AUTOCOMPLETE_SERVICE_URL=http://localhost:8000
PYTORCH_MODEL_PATH=/app/models
```

### Customization
- **Themes**: Modify `frontend/src/theme/` for custom themes
- **Colors**: Update accent colors in settings
- **Branding**: Replace logos and app icons
- **Features**: Enable/disable features via feature flags

## 🧪 Testing

### Backend Tests
```bash
cd backend
mix test
mix test --cover
```

### Frontend Tests
```bash
cd frontend
npm test
npm run test:coverage
```

### Autocomplete Tests
```bash
cd autocomplete
python -m pytest tests/
```

### Integration Tests
```bash
# Run full test suite
./deploy.sh docker deploy
npm run test:integration
```

## 📊 Monitoring & Analytics

### Health Checks
```bash
# Check service health
./deploy.sh docker health

# Individual service checks
curl http://localhost:4000/health
curl http://localhost:8000/autocomplete/health
```

### Metrics & Logging
- **Application Metrics**: Built-in Phoenix metrics
- **Database Monitoring**: PostgreSQL and ScyllaDB monitoring
- **Error Tracking**: Integrated error reporting
- **Performance Monitoring**: Request/response time tracking

## 🔒 Security

### Security Features
- **JWT Authentication** with refresh tokens
- **Password Hashing** with bcrypt
- **Rate Limiting** on API endpoints
- **CORS Protection** for web requests
- **Input Validation** and sanitization
- **SQL Injection Protection** via Ecto
- **XSS Prevention** with proper escaping

### Security Best Practices
- Regular security updates
- Environment variable protection
- Database connection encryption
- API endpoint authentication
- User data privacy compliance

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Code Style
- **Elixir**: Follow Elixir style guide with Credo
- **TypeScript**: Use ESLint and Prettier
- **Python**: Follow PEP 8 with Black formatter

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🆘 Support

- **Documentation**: Check our [docs](docs/) folder
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join our GitHub Discussions
- **Discord**: Join our community Discord server

## 🚧 Roadmap

### Upcoming Features
- **End-to-End Encryption** for messages
- **File Sharing** with cloud storage
- **Advanced Search** with filters
- **Message Threads** and replies
- **Custom Emojis** and reactions
- **Bot Integration** API
- **Mobile App Store** deployment
- **Desktop Applications** (Electron)

### Performance Improvements
- **Message Pagination** optimization
- **Image/Video Compression**
- **CDN Integration** for media
- **Database Sharding** for scale
- **WebSocket Connection** pooling

---

Built with ❤️ using Elixir, React Native, and Python
