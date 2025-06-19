import React, { useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { Provider as PaperProvider } from 'react-native-paper';
import { StatusBar } from 'expo-status-bar';
import { Platform, View } from 'react-native';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { useAuthStore } from './src/stores/authStore';
import { useChatStore } from './src/stores/chatStore';
import AuthNavigator from './src/navigation/AuthNavigator';
import MainNavigator from './src/navigation/MainNavigator';
import LoadingScreen from './src/components/LoadingScreen';
import { theme } from './src/theme';

// Import CSS for web platform
if (Platform.OS === 'web') {
    require('./app.css');
}

export default function App() {
    const { isAuthenticated, isLoading, loadStoredAuth } = useAuthStore();
    const { initialize, cleanup } = useChatStore();

    useEffect(() => {
        // Load stored authentication on app start
        loadStoredAuth();
    }, [loadStoredAuth]);

    useEffect(() => {
        if (isAuthenticated) {
            // Initialize chat store when authenticated
            initialize();
            return () => cleanup();
        }
    }, [isAuthenticated, initialize, cleanup]);

    if (isLoading) {
        return <LoadingScreen />;
    }

    return (
        <GestureHandlerRootView style={{ flex: 1 }}>
            <PaperProvider theme={theme}>
                <NavigationContainer>
                    <StatusBar
                        style="auto"
                        backgroundColor="transparent"
                        translucent={Platform.OS === 'android'}
                    />
                    <View style={{
                        flex: 1,
                        backgroundColor: theme.colors.background
                    }}>
                        {isAuthenticated ? <MainNavigator /> : <AuthNavigator />}
                    </View>
                </NavigationContainer>
            </PaperProvider>
        </GestureHandlerRootView>
    );
}
