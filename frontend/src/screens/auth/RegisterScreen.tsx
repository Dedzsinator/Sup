import React, { useState } from 'react';
import { View, StyleSheet, Alert } from 'react-native';
import { Text, TextInput, Button, Card } from 'react-native-paper';
import { StackNavigationProp } from '@react-navigation/stack';
import { useAuthStore } from '../../stores/authStore';
import { useTheme, Spacing } from '../../theme';
import { AuthStackParamList } from '../../navigation/AuthNavigator';

type RegisterScreenNavigationProp = StackNavigationProp<AuthStackParamList, 'Register'>;

interface Props {
    navigation: RegisterScreenNavigationProp;
}

export default function RegisterScreen({ navigation }: Props) {
    const [email, setEmail] = useState('');
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const { register, isLoading } = useAuthStore();
    const theme = useTheme();

    const handleRegister = async () => {
        if (!email.trim() || !username.trim() || !password.trim() || !confirmPassword.trim()) {
            Alert.alert('Error', 'Please fill in all fields');
            return;
        }

        if (password !== confirmPassword) {
            Alert.alert('Error', 'Passwords do not match');
            return;
        }

        if (password.length < 8) {
            Alert.alert('Error', 'Password must be at least 8 characters long');
            return;
        }

        const success = await register(email.trim(), username.trim(), password);
        if (!success) {
            Alert.alert('Error', 'Registration failed. Please try again.');
        }
    };

    return (
        <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
            <Card style={styles.card}>
                <Card.Content>
                    <Text variant="headlineMedium" style={styles.title}>
                        Create Account
                    </Text>
                    <Text variant="bodyLarge" style={styles.subtitle}>
                        Join the Sup community
                    </Text>

                    <TextInput
                        mode="outlined"
                        label="Email"
                        value={email}
                        onChangeText={setEmail}
                        keyboardType="email-address"
                        autoCapitalize="none"
                        style={styles.input}
                        disabled={isLoading}
                    />

                    <TextInput
                        mode="outlined"
                        label="Username"
                        value={username}
                        onChangeText={setUsername}
                        autoCapitalize="none"
                        style={styles.input}
                        disabled={isLoading}
                    />

                    <TextInput
                        mode="outlined"
                        label="Password"
                        value={password}
                        onChangeText={setPassword}
                        secureTextEntry
                        style={styles.input}
                        disabled={isLoading}
                    />

                    <TextInput
                        mode="outlined"
                        label="Confirm Password"
                        value={confirmPassword}
                        onChangeText={setConfirmPassword}
                        secureTextEntry
                        style={styles.input}
                        disabled={isLoading}
                    />

                    <Button
                        mode="contained"
                        onPress={handleRegister}
                        loading={isLoading}
                        disabled={isLoading}
                        style={styles.button}
                    >
                        Sign Up
                    </Button>

                    <Button
                        mode="text"
                        onPress={() => navigation.navigate('Login')}
                        disabled={isLoading}
                        style={styles.linkButton}
                    >
                        Already have an account? Sign in
                    </Button>
                </Card.Content>
            </Card>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        padding: Spacing.lg,
    },
    card: {
        padding: Spacing.lg,
    },
    title: {
        textAlign: 'center',
        marginBottom: Spacing.sm,
        fontWeight: 'bold',
    },
    subtitle: {
        textAlign: 'center',
        marginBottom: Spacing.xl,
        opacity: 0.7,
    },
    input: {
        marginBottom: Spacing.md,
    },
    button: {
        marginTop: Spacing.md,
        marginBottom: Spacing.sm,
    },
    linkButton: {
        marginTop: Spacing.sm,
    },
});
