import React, { useState } from 'react';
import { View, Alert, Dimensions, Platform, ScrollView, KeyboardAvoidingView, StyleSheet } from 'react-native';
import { Text } from 'react-native-paper';
import { StackNavigationProp } from '@react-navigation/stack';
import { Ionicons } from '@expo/vector-icons';
import { useAuthStore } from '../../stores/authStore';
import { useTheme, colors } from '../../theme';
import { AuthStackParamList } from '../../navigation/AuthNavigator';
import ModernCard from '../../components/ModernCard';
import ModernInput from '../../components/ModernInput';
import ModernButton from '../../components/ModernButton';

type LoginScreenNavigationProp = StackNavigationProp<AuthStackParamList, 'Login'>;

interface Props {
    navigation: LoginScreenNavigationProp;
}

const { width, height } = Dimensions.get('window');
const isTablet = width > 768;
const isDesktop = width > 1024;

export default function LoginScreen({ navigation }: Props) {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const { login, isLoading } = useAuthStore();
    const theme = useTheme();

    const handleLogin = async () => {
        if (!email.trim() || !password.trim()) {
            Alert.alert('Error', 'Please fill in all fields');
            return;
        }

        const success = await login(email.trim(), password);
        if (!success) {
            Alert.alert('Error', 'Invalid email or password');
        }
    };

    const containerMaxWidth = isDesktop ? 400 : isTablet ? 500 : width - 32;

    return (
        <KeyboardAvoidingView
            behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
            style={styles.container}
        >
            {/* Background */}
            <View style={[styles.background, { backgroundColor: colors.primary[50] }]} />

            {/* Animated Background Elements */}
            <View style={StyleSheet.absoluteFill}>
                <View style={[styles.bgElement1, { backgroundColor: colors.primary[200] }]} />
                <View style={[styles.bgElement2, { backgroundColor: colors.secondary[200] }]} />
                <View style={[styles.bgElement3, { backgroundColor: colors.accent[200] }]} />
            </View>

            <ScrollView
                contentContainerStyle={styles.scrollContent}
                showsVerticalScrollIndicator={false}
            >
                <View style={[styles.contentContainer, { maxWidth: containerMaxWidth }]}>
                    {/* Logo and Welcome Text */}
                    <View style={styles.headerContainer}>
                        <View style={[styles.logoContainer, {
                            backgroundColor: theme.colors.primary,
                            shadowColor: theme.colors.primary
                        }]}>
                            <Ionicons name="chatbubbles" size={32} color="white" />
                        </View>

                        <View style={styles.titleContainer}>
                            <Text style={[styles.title, { color: theme.colors.onBackground }]}>
                                Welcome to Sup
                            </Text>
                            <Text style={[styles.subtitle, { color: theme.colors.onBackground }]}>
                                Sign in to continue messaging
                            </Text>
                        </View>
                    </View>

                    {/* Login Form */}
                    <ModernCard variant="elevated" padding="lg" borderRadius="xl">
                        <ModernInput
                            label="Email Address"
                            value={email}
                            onChangeText={setEmail}
                            placeholder="Enter your email"
                            keyboardType="email-address"
                            autoCapitalize="none"
                            leftIcon={
                                <Ionicons
                                    name="mail-outline"
                                    size={20}
                                    color={theme.colors.onSurfaceVariant}
                                />
                            }
                        />

                        <ModernInput
                            label="Password"
                            value={password}
                            onChangeText={setPassword}
                            placeholder="Enter your password"
                            secureTextEntry={!showPassword}
                            leftIcon={
                                <Ionicons
                                    name="lock-closed-outline"
                                    size={20}
                                    color={theme.colors.onSurfaceVariant}
                                />
                            }
                            rightIcon={
                                <Ionicons
                                    name={showPassword ? "eye-off-outline" : "eye-outline"}
                                    size={20}
                                    color={theme.colors.onSurfaceVariant}
                                />
                            }
                            onRightIconPress={() => setShowPassword(!showPassword)}
                        />

                        <ModernButton
                            title="Sign In"
                            onPress={handleLogin}
                            loading={isLoading}
                            disabled={isLoading}
                            size="lg"
                            fullWidth
                        />

                        <View style={styles.dividerContainer}>
                            <View style={[styles.divider, { backgroundColor: theme.colors.outline }]} />
                            <Text style={[styles.dividerText, { color: theme.colors.onSurface }]}>
                                or
                            </Text>
                            <View style={[styles.divider, { backgroundColor: theme.colors.outline }]} />
                        </View>

                        <ModernButton
                            title="Don't have an account? Sign up"
                            onPress={() => navigation.navigate('Register')}
                            variant="ghost"
                            disabled={isLoading}
                            size="md"
                            fullWidth
                        />
                    </ModernCard>

                    {/* Footer */}
                    <View style={styles.footerContainer}>
                        <Text style={[styles.footerText, { color: theme.colors.onBackground }]}>
                            Secure • Private • Fast
                        </Text>
                    </View>
                </View>
            </ScrollView>
        </KeyboardAvoidingView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    background: {
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
    },
    bgElement1: {
        position: 'absolute',
        top: 40,
        right: 40,
        width: 128,
        height: 128,
        borderRadius: 64,
        opacity: 0.2,
    },
    bgElement2: {
        position: 'absolute',
        bottom: 80,
        left: 20,
        width: 96,
        height: 96,
        borderRadius: 48,
        opacity: 0.2,
    },
    bgElement3: {
        position: 'absolute',
        top: '33%',
        left: '25%',
        width: 64,
        height: 64,
        borderRadius: 32,
        opacity: 0.2,
    },
    scrollContent: {
        flexGrow: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 16,
    },
    contentContainer: {
        width: '100%',
    },
    headerContainer: {
        alignItems: 'center',
        marginBottom: 32,
    },
    logoContainer: {
        width: 80,
        height: 80,
        borderRadius: 40,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 24,
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.3,
        shadowRadius: 16,
        elevation: 8,
    },
    titleContainer: {
        alignItems: 'center',
    },
    title: {
        fontSize: 32,
        fontWeight: 'bold',
        textAlign: 'center',
        marginBottom: 8,
    },
    subtitle: {
        fontSize: 18,
        textAlign: 'center',
        opacity: 0.7,
    },
    dividerContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        marginVertical: 16,
    },
    divider: {
        flex: 1,
        height: 1,
        opacity: 0.5,
    },
    dividerText: {
        marginHorizontal: 16,
        fontSize: 14,
        opacity: 0.7,
    },
    footerContainer: {
        alignItems: 'center',
        marginTop: 32,
    },
    footerText: {
        fontSize: 14,
        opacity: 0.6,
        textAlign: 'center',
    },
});
