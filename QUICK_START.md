# Sup Messaging App - Quick Start Guide

## ✅ Setup Complete!

Both the frontend and backend are now properly configured and can be started.

### Prerequisites Met
- ✅ PostgreSQL installed and running
- ✅ ScyllaDB running in Docker
- ✅ Node.js and npm packages installed
- ✅ Elixir and Phoenix dependencies installed
- ✅ Database created and migrations run

## 🚀 Starting the Applications

### Backend (Elixir/Phoenix)
```bash
cd /home/deginandor/Documents/Programming/Sup/backend
mix run --no-halt
```

The backend will be available at:
- API: http://localhost:4000
- Health check: http://localhost:4000/health

### Frontend (React Native/Expo)
```bash
cd /home/deginandor/Documents/Programming/Sup/frontend
npm start
```

The frontend will be available at:
- Web: http://localhost:8081
- Mobile: Scan QR code with Expo Go app

## 📱 Access Points
- **Backend API**: http://localhost:4000
- **Frontend Web**: http://localhost:8081
- **Mobile**: Use Expo Go app to scan QR code

## 🛠️ Development Notes

### Backend
- Uses PostgreSQL for main database
- Uses ScyllaDB for message storage
- Redis dependency has been disabled for development
- All database migrations have been applied

### Frontend
- React Native with Expo
- Compatible with web, iOS, and Android
- All npm dependencies installed with legacy peer deps
- React DOM installed for web support

## 🔧 Troubleshooting

If you encounter issues:

1. **Backend won't start**: Check if PostgreSQL is running (`sudo systemctl status postgresql`)
2. **Frontend dependency issues**: Run `npm install --legacy-peer-deps`
3. **ScyllaDB connection issues**: Ensure Docker container is running (`docker ps`)

## 🎉 Ready to Code!

Both applications are now ready for development. You can start making changes to the code and see them reflected immediately.
