import React, { useState, useEffect, useCallback } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity, Switch, Alert } from 'react-native';
import { Text, Portal, Modal, Button, TextInput, Chip } from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import { useTheme, colors, Spacing } from '../../theme';
import ModernCard from '../../components/ModernCard';
import { apiClient } from '../../services/api';
import { useAuthStore } from '../../stores/authStore';
import { UserSettings } from '../../types';

interface ColorOption {
    name: string;
    value: string;
    color: string;
}

const ACCENT_COLORS: ColorOption[] = [
    { name: 'Blue', value: '#3B82F6', color: '#3B82F6' },
    { name: 'Purple', value: '#8B5CF6', color: '#8B5CF6' },
    { name: 'Pink', value: '#EC4899', color: '#EC4899' },
    { name: 'Green', value: '#10B981', color: '#10B981' },
    { name: 'Orange', value: '#F59E0B', color: '#F59E0B' },
    { name: 'Red', value: '#EF4444', color: '#EF4444' },
    { name: 'Teal', value: '#14B8A6', color: '#14B8A6' },
    { name: 'Indigo', value: '#6366F1', color: '#6366F1' },
];

const THEME_OPTIONS = [
    { label: 'System', value: 'system' },
    { label: 'Light', value: 'light' },
    { label: 'Dark', value: 'dark' },
];

const ACTIVITY_STATUS_OPTIONS = [
    { label: 'Online', value: 'online', icon: 'radio-button-on', color: colors.success[500] },
    { label: 'Away', value: 'away', icon: 'moon', color: colors.warning[500] },
    { label: 'Busy', value: 'busy', icon: 'do-not-disturb', color: colors.error[500] },
    { label: 'Invisible', value: 'invisible', icon: 'eye-off', color: colors.gray[500] },
];

const PRIVACY_OPTIONS = {
    online_status: [
        { label: 'Everyone', value: 'everyone' },
        { label: 'Friends Only', value: 'friends' },
        { label: 'Nobody', value: 'nobody' },
    ],
    profile_visibility: [
        { label: 'Everyone', value: 'everyone' },
        { label: 'Friends Only', value: 'friends' },
        { label: 'Nobody', value: 'nobody' },
    ],
};

export default function SettingsScreen() {
    const { user, updateProfile } = useAuthStore();
    const theme = useTheme();
    const [loading, setLoading] = useState(false);
    const [settings, setSettings] = useState<UserSettings | null>(null);

    // Modal states
    const [showAccentColorModal, setShowAccentColorModal] = useState(false);
    const [showThemeModal, setShowThemeModal] = useState(false);
    const [showActivityStatusModal, setShowActivityStatusModal] = useState(false);
    const [showPrivacyModal, setShowPrivacyModal] = useState(false);
    const [showProfileModal, setShowProfileModal] = useState(false);

    // Form states
    const [displayName, setDisplayName] = useState('');
    const [bio, setBio] = useState('');
    const [statusMessage, setStatusMessage] = useState('');

    const loadSettings = useCallback(async () => {
        try {
            setLoading(true);
            const response = await apiClient.getSettings();
            setSettings(response.data?.settings || null);

            // Initialize form fields
            if (user) {
                setDisplayName(user.display_name || '');
                setBio(user.bio || '');
                setStatusMessage(user.status_message || '');
            }
        } catch (error) {
            console.error('Failed to load settings:', error);
            Alert.alert('Error', 'Failed to load settings');
        } finally {
            setLoading(false);
        }
    }, [user]);

    useEffect(() => {
        loadSettings();
    }, [loadSettings]);

    const updateSettings = async (updates: Partial<UserSettings>) => {
        try {
            setLoading(true);
            await apiClient.updateSettings(updates);
            setSettings(prev => ({ ...prev!, ...updates }));
        } catch (error) {
            console.error('Failed to update settings:', error);
            Alert.alert('Error', 'Failed to update settings');
        } finally {
            setLoading(false);
        }
    };

    const updateUserProfile = async (updates: any) => {
        try {
            setLoading(true);
            await updateProfile(updates);
        } catch (error) {
            console.error('Failed to update profile:', error);
            Alert.alert('Error', 'Failed to update profile');
        } finally {
            setLoading(false);
        }
    };

    const updateNotificationSetting = (key: keyof UserSettings['notification_settings'], value: boolean) => {
        if (!settings) return;

        const updatedNotifications = {
            ...settings.notification_settings,
            [key]: value
        };

        updateSettings({ notification_settings: updatedNotifications });
    };

    const updatePrivacySetting = (key: keyof UserSettings['privacy_settings'], value: any) => {
        if (!settings) return;

        const updatedPrivacy = {
            ...settings.privacy_settings,
            [key]: value
        };

        updateSettings({ privacy_settings: updatedPrivacy });
    };

    const updateCallSetting = (key: keyof UserSettings['call_settings'], value: any) => {
        if (!settings) return;

        const updatedCall = {
            ...settings.call_settings,
            [key]: value
        };

        updateSettings({ call_settings: updatedCall });
    };

    const saveProfile = async () => {
        try {
            await updateUserProfile({
                display_name: displayName,
                bio: bio,
                status_message: statusMessage,
            });
            setShowProfileModal(false);
        } catch (error) {
            console.error('Failed to save profile:', error);
        }
    };

    const SettingItem = ({
        icon,
        title,
        subtitle,
        onPress,
        rightElement,
        iconBackgroundColor,
        iconColor
    }: {
        icon: string;
        title: string;
        subtitle?: string;
        onPress?: () => void;
        rightElement?: React.ReactNode;
        iconBackgroundColor: string;
        iconColor: string;
    }) => (
        <TouchableOpacity
            style={styles.settingItem}
            onPress={onPress}
            activeOpacity={0.7}
            disabled={!onPress}
        >
            <View style={[styles.settingIcon, { backgroundColor: iconBackgroundColor }]}>
                <Ionicons name={icon as any} size={20} color={iconColor} />
            </View>
            <View style={styles.settingContent}>
                <Text style={[styles.settingTitle, { color: theme.colors.onSurface }]}>
                    {title}
                </Text>
                {subtitle && (
                    <Text style={[styles.settingSubtitle, { color: theme.colors.onSurfaceVariant }]}>
                        {subtitle}
                    </Text>
                )}
            </View>
            {rightElement || (
                <Ionicons
                    name="chevron-forward"
                    size={20}
                    color={theme.colors.onSurfaceVariant}
                />
            )}
        </TouchableOpacity>
    );

    if (!settings) {
        return (
            <View style={[styles.container, styles.loading, { backgroundColor: theme.colors.background }]}>
                <Text style={{ color: theme.colors.onBackground }}>Loading settings...</Text>
            </View>
        );
    }

    return (
        <>
            <ScrollView
                style={[styles.container, { backgroundColor: theme.colors.background }]}
                contentContainerStyle={styles.scrollContent}
                showsVerticalScrollIndicator={false}
            >
                {/* Header */}
                <View style={styles.header}>
                    <Text style={[styles.headerTitle, { color: theme.colors.onBackground }]}>
                        Settings
                    </Text>
                </View>

                {/* Profile Section */}
                <ModernCard variant="elevated" padding="lg" borderRadius="xl" style={styles.section}>
                    <Text style={[styles.sectionTitle, { color: theme.colors.onSurface }]}>
                        Profile
                    </Text>

                    <SettingItem
                        icon="person"
                        title="Edit Profile"
                        subtitle="Display name, bio, status message"
                        iconBackgroundColor={colors.primary[100]}
                        iconColor={colors.primary[600]}
                        onPress={() => setShowProfileModal(true)}
                    />

                    <SettingItem
                        icon="radio-button-on"
                        title="Activity Status"
                        subtitle={ACTIVITY_STATUS_OPTIONS.find(opt => opt.value === user?.activity_status)?.label || 'Online'}
                        iconBackgroundColor={colors.success[100]}
                        iconColor={colors.success[600]}
                        onPress={() => setShowActivityStatusModal(true)}
                    />

                    <SettingItem
                        icon="color-palette"
                        title="Accent Color"
                        subtitle="Customize your theme color"
                        iconBackgroundColor={colors.secondary[100]}
                        iconColor={colors.secondary[600]}
                        onPress={() => setShowAccentColorModal(true)}
                        rightElement={
                            <View style={[styles.colorPreview, { backgroundColor: user?.accent_color }]} />
                        }
                    />

                    <SettingItem
                        icon="contrast"
                        title="Theme"
                        subtitle={THEME_OPTIONS.find(opt => opt.value === user?.theme_preference)?.label || 'System'}
                        iconBackgroundColor={colors.accent[100]}
                        iconColor={colors.accent[600]}
                        onPress={() => setShowThemeModal(true)}
                    />
                </ModernCard>

                {/* Notifications Section */}
                <ModernCard variant="elevated" padding="lg" borderRadius="xl" style={styles.section}>
                    <Text style={[styles.sectionTitle, { color: theme.colors.onSurface }]}>
                        Notifications
                    </Text>

                    <SettingItem
                        icon="notifications"
                        title="Messages"
                        subtitle="Notifications for new messages"
                        iconBackgroundColor={colors.primary[100]}
                        iconColor={colors.primary[600]}
                        rightElement={
                            <Switch
                                value={settings.notification_settings.messages}
                                onValueChange={(value) => updateNotificationSetting('messages', value)}
                                trackColor={{ false: colors.gray[300], true: colors.primary[200] }}
                                thumbColor={settings.notification_settings.messages ? colors.primary[500] : colors.gray[400]}
                            />
                        }
                    />

                    <SettingItem
                        icon="at"
                        title="Mentions"
                        subtitle="When someone mentions you"
                        iconBackgroundColor={colors.secondary[100]}
                        iconColor={colors.secondary[600]}
                        rightElement={
                            <Switch
                                value={settings.notification_settings.mentions}
                                onValueChange={(value) => updateNotificationSetting('mentions', value)}
                                trackColor={{ false: colors.gray[300], true: colors.secondary[200] }}
                                thumbColor={settings.notification_settings.mentions ? colors.secondary[500] : colors.gray[400]}
                            />
                        }
                    />

                    <SettingItem
                        icon="call"
                        title="Calls"
                        subtitle="Incoming voice and video calls"
                        iconBackgroundColor={colors.accent[100]}
                        iconColor={colors.accent[600]}
                        rightElement={
                            <Switch
                                value={settings.notification_settings.calls}
                                onValueChange={(value) => updateNotificationSetting('calls', value)}
                                trackColor={{ false: colors.gray[300], true: colors.accent[200] }}
                                thumbColor={settings.notification_settings.calls ? colors.accent[500] : colors.gray[400]}
                            />
                        }
                    />

                    <SettingItem
                        icon="volume-high"
                        title="Sound"
                        subtitle="Play notification sounds"
                        iconBackgroundColor={colors.warning[100]}
                        iconColor={colors.warning[600]}
                        rightElement={
                            <Switch
                                value={settings.notification_settings.sound}
                                onValueChange={(value) => updateNotificationSetting('sound', value)}
                                trackColor={{ false: colors.gray[300], true: colors.warning[200] }}
                                thumbColor={settings.notification_settings.sound ? colors.warning[500] : colors.gray[400]}
                            />
                        }
                    />
                </ModernCard>

                {/* Privacy & Security Section */}
                <ModernCard variant="elevated" padding="lg" borderRadius="xl" style={styles.section}>
                    <Text style={[styles.sectionTitle, { color: theme.colors.onSurface }]}>
                        Privacy & Security
                    </Text>

                    <SettingItem
                        icon="eye"
                        title="Privacy Settings"
                        subtitle="Control who can see your information"
                        iconBackgroundColor={colors.info[100]}
                        iconColor={colors.info[600]}
                        onPress={() => setShowPrivacyModal(true)}
                    />

                    <SettingItem
                        icon="checkmark-done"
                        title="Read Receipts"
                        subtitle="Show when you've read messages"
                        iconBackgroundColor={colors.success[100]}
                        iconColor={colors.success[600]}
                        rightElement={
                            <Switch
                                value={settings.privacy_settings.message_receipts}
                                onValueChange={(value) => updatePrivacySetting('message_receipts', value)}
                                trackColor={{ false: colors.gray[300], true: colors.success[200] }}
                                thumbColor={settings.privacy_settings.message_receipts ? colors.success[500] : colors.gray[400]}
                            />
                        }
                    />

                    <SettingItem
                        icon="create"
                        title="Typing Indicators"
                        subtitle="Show when you're typing"
                        iconBackgroundColor={colors.warning[100]}
                        iconColor={colors.warning[600]}
                        rightElement={
                            <Switch
                                value={settings.privacy_settings.typing_indicators}
                                onValueChange={(value) => updatePrivacySetting('typing_indicators', value)}
                                trackColor={{ false: colors.gray[300], true: colors.warning[200] }}
                                thumbColor={settings.privacy_settings.typing_indicators ? colors.warning[500] : colors.gray[400]}
                            />
                        }
                    />

                    <SettingItem
                        icon="lock-closed"
                        title="Two-Factor Authentication"
                        subtitle={user?.two_factor_enabled ? "Enabled" : "Disabled"}
                        iconBackgroundColor={colors.error[100]}
                        iconColor={colors.error[600]}
                        onPress={() => {/* Navigate to 2FA setup */ }}
                    />
                </ModernCard>

                {/* Voice & Video Section */}
                <ModernCard variant="elevated" padding="lg" borderRadius="xl" style={styles.section}>
                    <Text style={[styles.sectionTitle, { color: theme.colors.onSurface }]}>
                        Voice & Video
                    </Text>

                    <SettingItem
                        icon="videocam"
                        title="Camera Default"
                        subtitle="Start calls with camera on"
                        iconBackgroundColor={colors.primary[100]}
                        iconColor={colors.primary[600]}
                        rightElement={
                            <Switch
                                value={settings.call_settings.camera_default}
                                onValueChange={(value) => updateCallSetting('camera_default', value)}
                                trackColor={{ false: colors.gray[300], true: colors.primary[200] }}
                                thumbColor={settings.call_settings.camera_default ? colors.primary[500] : colors.gray[400]}
                            />
                        }
                    />

                    <SettingItem
                        icon="mic"
                        title="Microphone Default"
                        subtitle="Start calls with microphone on"
                        iconBackgroundColor={colors.secondary[100]}
                        iconColor={colors.secondary[600]}
                        rightElement={
                            <Switch
                                value={settings.call_settings.mic_default}
                                onValueChange={(value) => updateCallSetting('mic_default', value)}
                                trackColor={{ false: colors.gray[300], true: colors.secondary[200] }}
                                thumbColor={settings.call_settings.mic_default ? colors.secondary[500] : colors.gray[400]}
                            />
                        }
                    />

                    <SettingItem
                        icon="volume-off"
                        title="Noise Suppression"
                        subtitle="Reduce background noise"
                        iconBackgroundColor={colors.accent[100]}
                        iconColor={colors.accent[600]}
                        rightElement={
                            <Switch
                                value={settings.call_settings.noise_suppression}
                                onValueChange={(value) => updateCallSetting('noise_suppression', value)}
                                trackColor={{ false: colors.gray[300], true: colors.accent[200] }}
                                thumbColor={settings.call_settings.noise_suppression ? colors.accent[500] : colors.gray[400]}
                            />
                        }
                    />

                    <SettingItem
                        icon="trending-down"
                        title="Echo Cancellation"
                        subtitle="Prevent audio feedback"
                        iconBackgroundColor={colors.info[100]}
                        iconColor={colors.info[600]}
                        rightElement={
                            <Switch
                                value={settings.call_settings.echo_cancellation}
                                onValueChange={(value) => updateCallSetting('echo_cancellation', value)}
                                trackColor={{ false: colors.gray[300], true: colors.info[200] }}
                                thumbColor={settings.call_settings.echo_cancellation ? colors.info[500] : colors.gray[400]}
                            />
                        }
                    />
                </ModernCard>

                {/* Support Section */}
                <ModernCard variant="elevated" padding="lg" borderRadius="xl" style={styles.section}>
                    <Text style={[styles.sectionTitle, { color: theme.colors.onSurface }]}>
                        Support & Info
                    </Text>

                    <SettingItem
                        icon="help-circle"
                        title="Help & FAQ"
                        subtitle="Get help with common issues"
                        iconBackgroundColor={colors.info[100]}
                        iconColor={colors.info[600]}
                        onPress={() => {/* Navigate to help */ }}
                    />

                    <SettingItem
                        icon="document-text"
                        title="Terms of Service"
                        subtitle="Read our terms"
                        iconBackgroundColor={colors.secondary[100]}
                        iconColor={colors.secondary[600]}
                        onPress={() => {/* Navigate to terms */ }}
                    />

                    <SettingItem
                        icon="shield-checkmark"
                        title="Privacy Policy"
                        subtitle="Read our privacy policy"
                        iconBackgroundColor={colors.accent[100]}
                        iconColor={colors.accent[600]}
                        onPress={() => {/* Navigate to privacy policy */ }}
                    />

                    <SettingItem
                        icon="information-circle"
                        title="About"
                        subtitle="App version and info"
                        iconBackgroundColor={colors.gray[100]}
                        iconColor={colors.gray[600]}
                        onPress={() => {/* Navigate to about */ }}
                    />
                </ModernCard>

                {/* Footer spacing */}
                <View style={styles.footer} />
            </ScrollView>

            {/* Profile Edit Modal */}
            <Portal>
                <Modal
                    visible={showProfileModal}
                    onDismiss={() => setShowProfileModal(false)}
                    contentContainerStyle={[styles.modal, { backgroundColor: theme.colors.surface }]}
                >
                    <Text style={[styles.modalTitle, { color: theme.colors.onSurface }]}>
                        Edit Profile
                    </Text>

                    <TextInput
                        label="Display Name"
                        value={displayName}
                        onChangeText={setDisplayName}
                        style={styles.input}
                        mode="outlined"
                    />

                    <TextInput
                        label="Bio"
                        value={bio}
                        onChangeText={setBio}
                        style={styles.input}
                        mode="outlined"
                        multiline
                        numberOfLines={3}
                    />

                    <TextInput
                        label="Status Message"
                        value={statusMessage}
                        onChangeText={setStatusMessage}
                        style={styles.input}
                        mode="outlined"
                    />

                    <View style={styles.modalButtons}>
                        <Button
                            mode="outlined"
                            onPress={() => setShowProfileModal(false)}
                            style={styles.modalButton}
                        >
                            Cancel
                        </Button>
                        <Button
                            mode="contained"
                            onPress={saveProfile}
                            style={styles.modalButton}
                            loading={loading}
                        >
                            Save
                        </Button>
                    </View>
                </Modal>
            </Portal>

            {/* Accent Color Modal */}
            <Portal>
                <Modal
                    visible={showAccentColorModal}
                    onDismiss={() => setShowAccentColorModal(false)}
                    contentContainerStyle={[styles.modal, { backgroundColor: theme.colors.surface }]}
                >
                    <Text style={[styles.modalTitle, { color: theme.colors.onSurface }]}>
                        Choose Accent Color
                    </Text>

                    <View style={styles.colorGrid}>
                        {ACCENT_COLORS.map((color) => (
                            <TouchableOpacity
                                key={color.value}
                                style={[
                                    styles.colorOption,
                                    { backgroundColor: color.color },
                                    user?.accent_color === color.value && styles.selectedColor
                                ]}
                                onPress={() => {
                                    updateUserProfile({ accent_color: color.value });
                                    setShowAccentColorModal(false);
                                }}
                            >
                                {user?.accent_color === color.value && (
                                    <Ionicons name="checkmark" size={20} color="white" />
                                )}
                            </TouchableOpacity>
                        ))}
                    </View>
                </Modal>
            </Portal>

            {/* Theme Modal */}
            <Portal>
                <Modal
                    visible={showThemeModal}
                    onDismiss={() => setShowThemeModal(false)}
                    contentContainerStyle={[styles.modal, { backgroundColor: theme.colors.surface }]}
                >
                    <Text style={[styles.modalTitle, { color: theme.colors.onSurface }]}>
                        Choose Theme
                    </Text>

                    {THEME_OPTIONS.map((option) => (
                        <TouchableOpacity
                            key={option.value}
                            style={styles.optionItem}
                            onPress={() => {
                                updateUserProfile({ theme_preference: option.value });
                                setShowThemeModal(false);
                            }}
                        >
                            <Text style={[
                                styles.optionText,
                                { color: theme.colors.onSurface },
                                user?.theme_preference === option.value && { fontWeight: 'bold' }
                            ]}>
                                {option.label}
                            </Text>
                            {user?.theme_preference === option.value && (
                                <Ionicons name="checkmark" size={20} color={colors.primary[500]} />
                            )}
                        </TouchableOpacity>
                    ))}
                </Modal>
            </Portal>

            {/* Activity Status Modal */}
            <Portal>
                <Modal
                    visible={showActivityStatusModal}
                    onDismiss={() => setShowActivityStatusModal(false)}
                    contentContainerStyle={[styles.modal, { backgroundColor: theme.colors.surface }]}
                >
                    <Text style={[styles.modalTitle, { color: theme.colors.onSurface }]}>
                        Activity Status
                    </Text>

                    {ACTIVITY_STATUS_OPTIONS.map((option) => (
                        <TouchableOpacity
                            key={option.value}
                            style={styles.optionItem}
                            onPress={() => {
                                updateUserProfile({ activity_status: option.value });
                                setShowActivityStatusModal(false);
                            }}
                        >
                            <View style={styles.statusOption}>
                                <Ionicons name={option.icon as any} size={20} color={option.color} />
                                <Text style={[
                                    styles.optionText,
                                    { color: theme.colors.onSurface, marginLeft: Spacing.sm },
                                    user?.activity_status === option.value && { fontWeight: 'bold' }
                                ]}>
                                    {option.label}
                                </Text>
                            </View>
                            {user?.activity_status === option.value && (
                                <Ionicons name="checkmark" size={20} color={colors.primary[500]} />
                            )}
                        </TouchableOpacity>
                    ))}
                </Modal>
            </Portal>

            {/* Privacy Settings Modal */}
            <Portal>
                <Modal
                    visible={showPrivacyModal}
                    onDismiss={() => setShowPrivacyModal(false)}
                    contentContainerStyle={[styles.modal, { backgroundColor: theme.colors.surface }]}
                >
                    <Text style={[styles.modalTitle, { color: theme.colors.onSurface }]}>
                        Privacy Settings
                    </Text>

                    <View style={styles.privacySection}>
                        <Text style={[styles.privacyLabel, { color: theme.colors.onSurface }]}>
                            Who can see your online status
                        </Text>
                        <View style={styles.chipContainer}>
                            {PRIVACY_OPTIONS.online_status.map((option) => (
                                <Chip
                                    key={option.value}
                                    selected={settings.privacy_settings.online_status === option.value}
                                    onPress={() => updatePrivacySetting('online_status', option.value)}
                                    style={styles.chip}
                                >
                                    {option.label}
                                </Chip>
                            ))}
                        </View>
                    </View>

                    <View style={styles.privacySection}>
                        <Text style={[styles.privacyLabel, { color: theme.colors.onSurface }]}>
                            Who can see your profile
                        </Text>
                        <View style={styles.chipContainer}>
                            {PRIVACY_OPTIONS.profile_visibility.map((option) => (
                                <Chip
                                    key={option.value}
                                    selected={settings.privacy_settings.profile_visibility === option.value}
                                    onPress={() => updatePrivacySetting('profile_visibility', option.value)}
                                    style={styles.chip}
                                >
                                    {option.label}
                                </Chip>
                            ))}
                        </View>
                    </View>

                    <Button
                        mode="contained"
                        onPress={() => setShowPrivacyModal(false)}
                        style={styles.modalCloseButton}
                    >
                        Done
                    </Button>
                </Modal>
            </Portal>
        </>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    loading: {
        justifyContent: 'center',
        alignItems: 'center',
    },
    scrollContent: {
        paddingHorizontal: Spacing.lg,
    },
    header: {
        paddingTop: Spacing.lg,
        paddingBottom: Spacing.md,
    },
    headerTitle: {
        fontSize: 28,
        fontWeight: 'bold',
    },
    section: {
        marginBottom: Spacing.lg,
    },
    sectionTitle: {
        fontSize: 18,
        fontWeight: '600',
        marginBottom: Spacing.md,
    },
    settingItem: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingVertical: Spacing.md,
        borderBottomWidth: 1,
        borderBottomColor: 'rgba(0,0,0,0.05)',
    },
    settingIcon: {
        width: 36,
        height: 36,
        borderRadius: 18,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: Spacing.md,
    },
    settingContent: {
        flex: 1,
    },
    settingTitle: {
        fontSize: 16,
        fontWeight: '500',
        marginBottom: 2,
    },
    settingSubtitle: {
        fontSize: 14,
    },
    colorPreview: {
        width: 24,
        height: 24,
        borderRadius: 12,
        borderWidth: 2,
        borderColor: 'rgba(0,0,0,0.1)',
    },
    footer: {
        height: Spacing.xl,
    },
    modal: {
        margin: Spacing.xl,
        padding: Spacing.lg,
        borderRadius: 16,
    },
    modalTitle: {
        fontSize: 20,
        fontWeight: 'bold',
        marginBottom: Spacing.lg,
    },
    input: {
        marginBottom: Spacing.md,
    },
    modalButtons: {
        flexDirection: 'row',
        justifyContent: 'flex-end',
        marginTop: Spacing.lg,
    },
    modalButton: {
        marginLeft: Spacing.sm,
    },
    colorGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        justifyContent: 'space-between',
    },
    colorOption: {
        width: 50,
        height: 50,
        borderRadius: 25,
        marginBottom: Spacing.md,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 2,
        borderColor: 'transparent',
    },
    selectedColor: {
        borderColor: 'rgba(255,255,255,0.5)',
    },
    optionItem: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingVertical: Spacing.md,
        borderBottomWidth: 1,
        borderBottomColor: 'rgba(0,0,0,0.05)',
    },
    optionText: {
        fontSize: 16,
    },
    statusOption: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    privacySection: {
        marginBottom: Spacing.lg,
    },
    privacyLabel: {
        fontSize: 16,
        fontWeight: '500',
        marginBottom: Spacing.sm,
    },
    chipContainer: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: Spacing.sm,
    },
    chip: {
        marginRight: Spacing.xs,
        marginBottom: Spacing.xs,
    },
    modalCloseButton: {
        marginTop: Spacing.lg,
    },
});
