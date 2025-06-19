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

type RegisterScreenNavigationProp = StackNavigationProp<AuthStackParamList, 'Register'>;

interface Props {
    navigation: RegisterScreenNavigationProp;
}

const { width, height } = Dimensions.get('window');
const isTablet = width > 768;
const isDesktop = width > 1024;

export default function RegisterScreen({ navigation }: Props) {
    const [email, setEmail] = useState('');
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);
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

    const containerMaxWidth = isDesktop ? 400 : isTablet ? 500 : width - 32;

    return (
        <KeyboardAvoidingView
            behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
            style={styles.container}
        >
            {/* Background */}
            <View style={[styles.background, { backgroundColor: colors.secondary[50] }]} />

            {/* Animated Background Elements */}
            <View style={StyleSheet.absoluteFill}>
                <View style={[styles.bgElement1, { backgroundColor: colors.secondary[200] }]} />
                <View style={[styles.bgElement2, { backgroundColor: colors.accent[200] }]} />
                <View style={[styles.bgElement3, { backgroundColor: colors.primary[200] }]} />
            </View>

            <ScrollView
                contentContainerStyle={styles.scrollContent}
                showsVerticalScrollIndicator={false}
            >
                <View style={[styles.contentContainer, { maxWidth: containerMaxWidth }]}>
                    {/* Header */}
                    <View style={styles.headerContainer}>
                        <View style={[styles.logoContainer, {
                            backgroundColor: theme.colors.secondary,
                            shadowColor: theme.colors.secondary
                        }]}>
                            <Ionicons name="person-add" size={32} color="white" />
                        </View>

                        <View style={styles.titleContainer}>
                            <Text style={[styles.title, { color: theme.colors.onBackground }]}>
                                Create Account
                            </Text>
                            <Text style={[styles.subtitle, { color: theme.colors.onBackground }]}>
                                Join the conversation
                            </Text>
                        </View>
                    </View>

                    {/* Registration Form */}
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
                            label="Username"
                            value={username}
                            onChangeText={setUsername}
                            placeholder="Choose a username"
                            autoCapitalize="none"
                            leftIcon={
                                <Ionicons
                                    name="person-outline"
                                    size={20}
                                    color={theme.colors.onSurfaceVariant}
                                />
                            }
                        />

                        <ModernInput
                            label="Password"
                            value={password}
                            onChangeText={setPassword}
                            placeholder="Create a password"
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

                        <ModernInput
                            label="Confirm Password"
                            value={confirmPassword}
                            onChangeText={setConfirmPassword}
                            placeholder="Confirm your password"
                            secureTextEntry={!showConfirmPassword}
                            leftIcon={
                                <Ionicons
                                    name="lock-closed-outline"
                                    size={20}
                                    color={theme.colors.onSurfaceVariant}
                                />
                            }
                            rightIcon={
                                <Ionicons
                                    name={showConfirmPassword ? "eye-off-outline" : "eye-outline"}
                                    size={20}
                                    color={theme.colors.onSurfaceVariant}
                                />
                            }
                            onRightIconPress={() => setShowConfirmPassword(!showConfirmPassword)}
                            error={confirmPassword && password !== confirmPassword ? "Passwords don't match" : undefined}
                        />

                        {/* Password Requirements */}
                        <View style={styles.requirementsContainer}>
                            <Text style={[styles.requirementsTitle, { color: theme.colors.onSurface }]}>
                                Password Requirements:
                            </Text>
                            <View style={styles.requirementItem}>
                                <Ionicons
                                    name={password.length >= 8 ? "checkmark-circle" : "ellipse-outline"}
                                    size={16}
                                    color={password.length >= 8 ? colors.accent[500] : theme.colors.onSurfaceVariant}
                                />
                                <Text style={[styles.requirementText, {
                                    color: password.length >= 8 ? colors.accent[500] : theme.colors.onSurfaceVariant
                                }]}>
                                    At least 8 characters
                                </Text>
                            </View>
                        </View>

                        <ModernButton
                            title="Create Account"
                            onPress={handleRegister}
                            loading={isLoading}
                            disabled={isLoading}
                            size="lg"
                            fullWidth
                            variant="secondary"
                        />

                        <View style={styles.dividerContainer}>
                            <View style={[styles.divider, { backgroundColor: theme.colors.outline }]} />
                            <Text style={[styles.dividerText, { color: theme.colors.onSurface }]}>
                                or
                            </Text>
                            <View style={[styles.divider, { backgroundColor: theme.colors.outline }]} />
                        </View>

                        <ModernButton
                            title="Already have an account? Sign in"
                            onPress={() => navigation.navigate('Login')}
                            variant="ghost"
                            disabled={isLoading}
                            size="md"
                            fullWidth
                        />
                    </ModernCard>

                    {/* Footer */}
                    <View style={styles.footerContainer}>
                        <Text style={[styles.footerText, { color: theme.colors.onBackground }]}>
                            By creating an account, you agree to our Terms of Service
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
        top: 60,
        left: 30,
        width: 120,
        height: 120,
        borderRadius: 60,
        opacity: 0.2,
    },
    bgElement2: {
        position: 'absolute',
        bottom: 100,
        right: 30,
        width: 100,
        height: 100,
        borderRadius: 50,
        opacity: 0.2,
    },
    bgElement3: {
        position: 'absolute',
        top: '40%',
        right: '20%',
        width: 80,
        height: 80,
        borderRadius: 40,
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
    requirementsContainer: {
        marginBottom: 16,
        padding: 12,
        borderRadius: 8,
        backgroundColor: 'rgba(0, 0, 0, 0.02)',
    },
    requirementsTitle: {
        fontSize: 14,
        fontWeight: '600',
        marginBottom: 8,
    },
    requirementItem: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    requirementText: {
        fontSize: 13,
        marginLeft: 8,
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
        marginTop: 24,
    },
    footerText: {
        fontSize: 12,
        opacity: 0.6,
        textAlign: 'center',
        paddingHorizontal: 20,
    },
});
