import React, { useState } from 'react';
import { View, Alert, Dimensions, Platform, ScrollView, KeyboardAvoidingView, StyleSheet, TouchableOpacity, Image } from 'react-native';
import { Text } from 'react-native-paper';
import { StackNavigationProp } from '@react-navigation/stack';
import { Ionicons } from '@expo/vector-icons';
import { useAuthStore } from '../../stores/authStore';
import { useTheme, colors, Spacing } from '../../theme';
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

const STEP_ACCOUNT = 0;
const STEP_PROFILE = 1;
const STEP_VERIFICATION = 2;

export default function RegisterScreen({ navigation }: Props) {
    const [currentStep, setCurrentStep] = useState(STEP_ACCOUNT);

    // Account step
    const [email, setEmail] = useState('');
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);

    // Profile step
    const [displayName, setDisplayName] = useState('');
    const [bio, setBio] = useState('');
    const [statusMessage, setStatusMessage] = useState('Hey there! I am using Sup.');
    const [avatarUri, setAvatarUri] = useState<string | null>(null);
    const [accentColor, setAccentColor] = useState('#3B82F6');

    const { register, isLoading } = useAuthStore();
    const theme = useTheme();

    const accentColors = [
        '#3B82F6', '#EF4444', '#10B981', '#F59E0B',
        '#8B5CF6', '#EC4899', '#06B6D4', '#84CC16'
    ];

    const validateAccountStep = () => {
        if (!email.trim() || !username.trim() || !password.trim() || !confirmPassword.trim()) {
            Alert.alert('Error', 'Please fill in all fields');
            return false;
        }

        if (password !== confirmPassword) {
            Alert.alert('Error', 'Passwords do not match');
            return false;
        }

        if (password.length < 8) {
            Alert.alert('Error', 'Password must be at least 8 characters long');
            return false;
        }

        if (!/^[^\s]+@[^\s]+\.[^\s]+$/.test(email)) {
            Alert.alert('Error', 'Please enter a valid email address');
            return false;
        }

        if (username.length < 3) {
            Alert.alert('Error', 'Username must be at least 3 characters long');
            return false;
        }

        return true;
    };

    const handleNextStep = () => {
        if (currentStep === STEP_ACCOUNT) {
            if (validateAccountStep()) {
                setDisplayName(username); // Default display name to username
                setCurrentStep(STEP_PROFILE);
            }
        } else if (currentStep === STEP_PROFILE) {
            setCurrentStep(STEP_VERIFICATION);
        }
    };

    const handlePrevStep = () => {
        if (currentStep > STEP_ACCOUNT) {
            setCurrentStep(currentStep - 1);
        }
    };

    const handleRegister = async () => {
        const success = await register(email.trim(), username.trim(), password, {
            display_name: displayName.trim() || username.trim(),
            bio: bio.trim(),
            status_message: statusMessage.trim(),
            avatar_url: avatarUri,
            accent_color: accentColor
        });

        if (!success) {
            Alert.alert('Error', 'Registration failed. Please try again.');
        }
    };

    const showImagePicker = () => {
        Alert.alert(
            'Profile Picture',
            'Choose how you\'d like to set your profile picture',
            [
                { text: 'Camera', onPress: () => { } },
                { text: 'Photo Library', onPress: () => { } },
                { text: 'Cancel', style: 'cancel' }
            ]
        );
    };

    const containerMaxWidth = isDesktop ? 480 : isTablet ? 500 : width - 32;

    const renderStepIndicator = () => (
        <View style={styles.stepIndicator}>
            {[STEP_ACCOUNT, STEP_PROFILE, STEP_VERIFICATION].map((step) => (
                <View key={step} style={styles.stepIndicatorContainer}>
                    <View style={[
                        styles.stepDot,
                        {
                            backgroundColor: currentStep >= step ? theme.colors.primary : theme.colors.outline
                        }
                    ]}>
                        {currentStep > step && (
                            <Ionicons name="checkmark" size={12} color="white" />
                        )}
                    </View>
                    {step < STEP_VERIFICATION && (
                        <View style={[
                            styles.stepLine,
                            {
                                backgroundColor: currentStep > step ? theme.colors.primary : theme.colors.outline
                            }
                        ]} />
                    )}
                </View>
            ))}
        </View>
    );

    const renderAccountStep = () => (
        <View>
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
        </View>
    );

    const renderProfileStep = () => (
        <View>
            {/* Avatar Selection */}
            <View style={styles.avatarSection}>
                <Text style={[styles.sectionTitle, { color: theme.colors.onSurface }]}>
                    Profile Picture
                </Text>
                <TouchableOpacity style={styles.avatarContainer} onPress={showImagePicker}>
                    {avatarUri ? (
                        <Image source={{ uri: avatarUri }} style={styles.avatar} />
                    ) : (
                        <View style={[styles.avatarPlaceholder, { backgroundColor: theme.colors.surfaceVariant }]}>
                            <Ionicons name="camera" size={32} color={theme.colors.onSurfaceVariant} />
                        </View>
                    )}
                    <View style={[styles.avatarEditIcon, { backgroundColor: theme.colors.primary }]}>
                        <Ionicons name="pencil" size={16} color="white" />
                    </View>
                </TouchableOpacity>
            </View>

            <ModernInput
                label="Display Name"
                value={displayName}
                onChangeText={setDisplayName}
                placeholder="How should others see you?"
                leftIcon={
                    <Ionicons
                        name="person"
                        size={20}
                        color={theme.colors.onSurfaceVariant}
                    />
                }
            />

            <ModernInput
                label="Bio (Optional)"
                value={bio}
                onChangeText={setBio}
                placeholder="Tell others about yourself"
                multiline
                numberOfLines={3}
                leftIcon={
                    <Ionicons
                        name="document-text-outline"
                        size={20}
                        color={theme.colors.onSurfaceVariant}
                    />
                }
            />

            <ModernInput
                label="Status Message"
                value={statusMessage}
                onChangeText={setStatusMessage}
                placeholder="What's your status?"
                leftIcon={
                    <Ionicons
                        name="chatbubble-outline"
                        size={20}
                        color={theme.colors.onSurfaceVariant}
                    />
                }
            />

            {/* Accent Color Selection */}
            <View style={styles.colorSection}>
                <Text style={[styles.sectionTitle, { color: theme.colors.onSurface }]}>
                    Choose Your Theme Color
                </Text>
                <View style={styles.colorGrid}>
                    {accentColors.map((color) => (
                        <TouchableOpacity
                            key={color}
                            style={[
                                styles.colorOption,
                                { backgroundColor: color },
                                accentColor === color && styles.selectedColor
                            ]}
                            onPress={() => setAccentColor(color)}
                        >
                            {accentColor === color && (
                                <Ionicons name="checkmark" size={20} color="white" />
                            )}
                        </TouchableOpacity>
                    ))}
                </View>
            </View>
        </View>
    );

    const renderVerificationStep = () => (
        <View style={styles.verificationContainer}>
            <View style={styles.verificationIcon}>
                <Ionicons name="mail-outline" size={48} color={theme.colors.primary} />
            </View>
            <Text style={[styles.verificationTitle, { color: theme.colors.onSurface }]}>
                Almost Done!
            </Text>
            <Text style={[styles.verificationText, { color: theme.colors.onSurfaceVariant }]}>
                We'll send you a verification email to confirm your account.
                You can start using Sup right away, but some features will be limited until you verify.
            </Text>

            <View style={styles.summaryCard}>
                <View style={styles.summaryRow}>
                    <Text style={[styles.summaryLabel, { color: theme.colors.onSurfaceVariant }]}>Email:</Text>
                    <Text style={[styles.summaryValue, { color: theme.colors.onSurface }]}>{email}</Text>
                </View>
                <View style={styles.summaryRow}>
                    <Text style={[styles.summaryLabel, { color: theme.colors.onSurfaceVariant }]}>Username:</Text>
                    <Text style={[styles.summaryValue, { color: theme.colors.onSurface }]}>{username}</Text>
                </View>
                <View style={styles.summaryRow}>
                    <Text style={[styles.summaryLabel, { color: theme.colors.onSurfaceVariant }]}>Display Name:</Text>
                    <Text style={[styles.summaryValue, { color: theme.colors.onSurface }]}>{displayName || username}</Text>
                </View>
            </View>
        </View>
    );

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
                                {currentStep === STEP_ACCOUNT && 'Create Account'}
                                {currentStep === STEP_PROFILE && 'Set Up Profile'}
                                {currentStep === STEP_VERIFICATION && 'Ready to Go!'}
                            </Text>
                            <Text style={[styles.subtitle, { color: theme.colors.onBackground }]}>
                                {currentStep === STEP_ACCOUNT && 'Join the conversation'}
                                {currentStep === STEP_PROFILE && 'Make it yours'}
                                {currentStep === STEP_VERIFICATION && 'Welcome to Sup'}
                            </Text>
                        </View>
                    </View>

                    {/* Step Indicator */}
                    {renderStepIndicator()}

                    {/* Registration Form */}
                    <ModernCard variant="elevated" padding="lg" borderRadius="xl">
                        {currentStep === STEP_ACCOUNT && renderAccountStep()}
                        {currentStep === STEP_PROFILE && renderProfileStep()}
                        {currentStep === STEP_VERIFICATION && renderVerificationStep()}

                        {/* Navigation Buttons */}
                        <View style={styles.navigationButtons}>
                            {currentStep > STEP_ACCOUNT && (
                                <ModernButton
                                    title="Back"
                                    onPress={handlePrevStep}
                                    variant="ghost"
                                    size="lg"
                                />
                            )}

                            {currentStep < STEP_VERIFICATION ? (
                                <ModernButton
                                    title="Next"
                                    onPress={handleNextStep}
                                    size="lg"
                                />
                            ) : (
                                <ModernButton
                                    title="Create Account"
                                    onPress={handleRegister}
                                    loading={isLoading}
                                    disabled={isLoading}
                                    size="lg"
                                />
                            )}
                        </View>

                        {currentStep === STEP_ACCOUNT && (
                            <>
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
                            </>
                        )}
                    </ModernCard>

                    {/* Footer */}
                    <View style={styles.footerContainer}>
                        <Text style={[styles.footerText, { color: theme.colors.onBackground }]}>
                            By creating an account, you agree to our Terms of Service and Privacy Policy
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
        backgroundColor: colors.secondary[50],
    },
    background: {
        ...StyleSheet.absoluteFillObject,
    },
    bgElement1: {
        position: 'absolute',
        top: -50,
        right: -50,
        width: 150,
        height: 150,
        borderRadius: 75,
        opacity: 0.3,
    },
    bgElement2: {
        position: 'absolute',
        top: '30%',
        left: -30,
        width: 100,
        height: 100,
        borderRadius: 50,
        opacity: 0.2,
    },
    bgElement3: {
        position: 'absolute',
        bottom: '10%',
        right: -20,
        width: 80,
        height: 80,
        borderRadius: 40,
        opacity: 0.25,
    },
    scrollContent: {
        flexGrow: 1,
        justifyContent: 'center',
        paddingHorizontal: Spacing.lg,
        paddingVertical: Spacing.xl,
    },
    contentContainer: {
        width: '100%',
        alignSelf: 'center',
    },
    headerContainer: {
        alignItems: 'center',
        marginBottom: Spacing.xl,
    },
    logoContainer: {
        width: 80,
        height: 80,
        borderRadius: 40,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: Spacing.lg,
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 8,
        elevation: 8,
    },
    titleContainer: {
        alignItems: 'center',
    },
    title: {
        fontSize: 28,
        fontWeight: 'bold',
        textAlign: 'center',
        marginBottom: Spacing.xs,
    },
    subtitle: {
        fontSize: 16,
        textAlign: 'center',
        opacity: 0.8,
    },
    stepIndicator: {
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: Spacing.xl,
    },
    stepIndicatorContainer: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    stepDot: {
        width: 32,
        height: 32,
        borderRadius: 16,
        alignItems: 'center',
        justifyContent: 'center',
    },
    stepLine: {
        width: 40,
        height: 2,
        marginHorizontal: Spacing.xs,
    },
    requirementsContainer: {
        marginTop: Spacing.md,
        padding: Spacing.md,
        backgroundColor: 'rgba(59, 130, 246, 0.05)',
        borderRadius: 12,
        borderWidth: 1,
        borderColor: 'rgba(59, 130, 246, 0.2)',
    },
    requirementsTitle: {
        fontSize: 14,
        fontWeight: '600',
        marginBottom: Spacing.sm,
    },
    requirementItem: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: Spacing.xs,
    },
    requirementText: {
        fontSize: 14,
        marginLeft: Spacing.sm,
    },
    avatarSection: {
        alignItems: 'center',
        marginBottom: Spacing.xl,
    },
    sectionTitle: {
        fontSize: 18,
        fontWeight: '600',
        marginBottom: Spacing.lg,
    },
    avatarContainer: {
        position: 'relative',
        marginBottom: Spacing.lg,
    },
    avatar: {
        width: 120,
        height: 120,
        borderRadius: 60,
    },
    avatarPlaceholder: {
        width: 120,
        height: 120,
        borderRadius: 60,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 2,
        borderColor: 'rgba(0,0,0,0.1)',
        borderStyle: 'dashed',
    },
    avatarEditIcon: {
        position: 'absolute',
        bottom: 0,
        right: 0,
        width: 36,
        height: 36,
        borderRadius: 18,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 3,
        borderColor: 'white',
    },
    colorSection: {
        marginTop: Spacing.lg,
    },
    colorGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        justifyContent: 'space-between',
        marginTop: Spacing.md,
    },
    colorOption: {
        width: 44,
        height: 44,
        borderRadius: 22,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: Spacing.md,
        borderWidth: 3,
        borderColor: 'transparent',
    },
    selectedColor: {
        borderColor: 'white',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.25,
        shadowRadius: 4,
        elevation: 4,
    },
    verificationContainer: {
        alignItems: 'center',
        paddingVertical: Spacing.xl,
    },
    verificationIcon: {
        marginBottom: Spacing.lg,
    },
    verificationTitle: {
        fontSize: 24,
        fontWeight: 'bold',
        textAlign: 'center',
        marginBottom: Spacing.md,
    },
    verificationText: {
        fontSize: 16,
        textAlign: 'center',
        lineHeight: 24,
        marginBottom: Spacing.xl,
    },
    summaryCard: {
        width: '100%',
        backgroundColor: 'rgba(59, 130, 246, 0.05)',
        borderRadius: 12,
        padding: Spacing.lg,
        borderWidth: 1,
        borderColor: 'rgba(59, 130, 246, 0.2)',
    },
    summaryRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingVertical: Spacing.sm,
    },
    summaryLabel: {
        fontSize: 14,
        fontWeight: '500',
    },
    summaryValue: {
        fontSize: 14,
        fontWeight: '600',
    },
    navigationButtons: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginTop: Spacing.xl,
    },
    dividerContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        marginVertical: Spacing.lg,
    },
    divider: {
        flex: 1,
        height: 1,
    },
    dividerText: {
        paddingHorizontal: Spacing.md,
        fontSize: 14,
        opacity: 0.7,
    },
    footerContainer: {
        marginTop: Spacing.xl,
        alignItems: 'center',
    },
    footerText: {
        fontSize: 12,
        textAlign: 'center',
        opacity: 0.7,
        lineHeight: 18,
    },
});
