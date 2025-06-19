import React, { useState } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity, Switch } from 'react-native';
import { Text } from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import { useTheme, colors, Spacing } from '../../theme';
import ModernCard from '../../components/ModernCard';

export default function SettingsScreen() {
    const [notificationsEnabled, setNotificationsEnabled] = useState(true);
    const [soundEnabled, setSoundEnabled] = useState(true);
    const [vibrationEnabled, setVibrationEnabled] = useState(true);
    const [darkMode, setDarkMode] = useState(false);
    const [onlineStatus, setOnlineStatus] = useState(true);
    const [readReceipts, setReadReceipts] = useState(true);
    const [typingIndicators, setTypingIndicators] = useState(true);
    const theme = useTheme();

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

    return (
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

            {/* Notifications Section */}
            <ModernCard variant="elevated" padding="lg" borderRadius="xl" style={styles.section}>
                <Text style={[styles.sectionTitle, { color: theme.colors.onSurface }]}>
                    Notifications
                </Text>

                <SettingItem
                    icon="notifications"
                    title="Push Notifications"
                    subtitle="Receive notifications for new messages"
                    iconBackgroundColor={colors.primary[100]}
                    iconColor={colors.primary[600]}
                    rightElement={
                        <Switch
                            value={notificationsEnabled}
                            onValueChange={setNotificationsEnabled}
                            trackColor={{ false: colors.gray[300], true: colors.primary[200] }}
                            thumbColor={notificationsEnabled ? colors.primary[500] : colors.gray[400]}
                        />
                    }
                />

                <SettingItem
                    icon="volume-high"
                    title="Sound"
                    subtitle="Play sound for notifications"
                    iconBackgroundColor={colors.secondary[100]}
                    iconColor={colors.secondary[600]}
                    rightElement={
                        <Switch
                            value={soundEnabled}
                            onValueChange={setSoundEnabled}
                            disabled={!notificationsEnabled}
                            trackColor={{ false: colors.gray[300], true: colors.secondary[200] }}
                            thumbColor={soundEnabled && notificationsEnabled ? colors.secondary[500] : colors.gray[400]}
                        />
                    }
                />

                <SettingItem
                    icon="phone-portrait"
                    title="Vibration"
                    subtitle="Vibrate for notifications"
                    iconBackgroundColor={colors.accent[100]}
                    iconColor={colors.accent[600]}
                    rightElement={
                        <Switch
                            value={vibrationEnabled}
                            onValueChange={setVibrationEnabled}
                            disabled={!notificationsEnabled}
                            trackColor={{ false: colors.gray[300], true: colors.accent[200] }}
                            thumbColor={vibrationEnabled && notificationsEnabled ? colors.accent[500] : colors.gray[400]}
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
                    icon="radio-button-on"
                    title="Online Status"
                    subtitle="Show when you're online"
                    iconBackgroundColor={colors.success[100]}
                    iconColor={colors.success[600]}
                    rightElement={
                        <Switch
                            value={onlineStatus}
                            onValueChange={setOnlineStatus}
                            trackColor={{ false: colors.gray[300], true: colors.success[200] }}
                            thumbColor={onlineStatus ? colors.success[500] : colors.gray[400]}
                        />
                    }
                />

                <SettingItem
                    icon="checkmark-done"
                    title="Read Receipts"
                    subtitle="Show when you've read messages"
                    iconBackgroundColor={colors.info[100]}
                    iconColor={colors.info[600]}
                    rightElement={
                        <Switch
                            value={readReceipts}
                            onValueChange={setReadReceipts}
                            trackColor={{ false: colors.gray[300], true: colors.info[200] }}
                            thumbColor={readReceipts ? colors.info[500] : colors.gray[400]}
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
                            value={typingIndicators}
                            onValueChange={setTypingIndicators}
                            trackColor={{ false: colors.gray[300], true: colors.warning[200] }}
                            thumbColor={typingIndicators ? colors.warning[500] : colors.gray[400]}
                        />
                    }
                />

                <SettingItem
                    icon="lock-closed"
                    title="Block List"
                    subtitle="Manage blocked users"
                    iconBackgroundColor={colors.error[100]}
                    iconColor={colors.error[600]}
                    onPress={() => { }}
                />
            </ModernCard>

            {/* Chat Settings Section */}
            <ModernCard variant="elevated" padding="lg" borderRadius="xl" style={styles.section}>
                <Text style={[styles.sectionTitle, { color: theme.colors.onSurface }]}>
                    Chat Settings
                </Text>

                <SettingItem
                    icon="text"
                    title="Font Size"
                    subtitle="Adjust message text size"
                    iconBackgroundColor={colors.primary[100]}
                    iconColor={colors.primary[600]}
                    onPress={() => { }}
                />

                <SettingItem
                    icon="color-palette"
                    title="Theme"
                    subtitle="Choose app appearance"
                    iconBackgroundColor={colors.secondary[100]}
                    iconColor={colors.secondary[600]}
                    rightElement={
                        <Switch
                            value={darkMode}
                            onValueChange={setDarkMode}
                            trackColor={{ false: colors.gray[300], true: colors.secondary[200] }}
                            thumbColor={darkMode ? colors.secondary[500] : colors.gray[400]}
                        />
                    }
                />

                <SettingItem
                    icon="language"
                    title="Language"
                    subtitle="Change app language"
                    iconBackgroundColor={colors.accent[100]}
                    iconColor={colors.accent[600]}
                    onPress={() => { }}
                />

                <SettingItem
                    icon="image"
                    title="Media Auto-Download"
                    subtitle="Automatically download media"
                    iconBackgroundColor={colors.info[100]}
                    iconColor={colors.info[600]}
                    onPress={() => { }}
                />
            </ModernCard>

            {/* Storage Section */}
            <ModernCard variant="elevated" padding="lg" borderRadius="xl" style={styles.section}>
                <Text style={[styles.sectionTitle, { color: theme.colors.onSurface }]}>
                    Storage & Data
                </Text>

                <SettingItem
                    icon="server"
                    title="Storage Usage"
                    subtitle="View storage breakdown"
                    iconBackgroundColor={colors.warning[100]}
                    iconColor={colors.warning[600]}
                    onPress={() => { }}
                />

                <SettingItem
                    icon="trash"
                    title="Clear Cache"
                    subtitle="Free up storage space"
                    iconBackgroundColor={colors.error[100]}
                    iconColor={colors.error[600]}
                    onPress={() => { }}
                />

                <SettingItem
                    icon="download"
                    title="Export Data"
                    subtitle="Download your data"
                    iconBackgroundColor={colors.success[100]}
                    iconColor={colors.success[600]}
                    onPress={() => { }}
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
                    onPress={() => { }}
                />

                <SettingItem
                    icon="mail"
                    title="Contact Support"
                    subtitle="Send us a message"
                    iconBackgroundColor={colors.primary[100]}
                    iconColor={colors.primary[600]}
                    onPress={() => { }}
                />

                <SettingItem
                    icon="document-text"
                    title="Terms of Service"
                    subtitle="Read our terms"
                    iconBackgroundColor={colors.secondary[100]}
                    iconColor={colors.secondary[600]}
                    onPress={() => { }}
                />

                <SettingItem
                    icon="shield-checkmark"
                    title="Privacy Policy"
                    subtitle="Read our privacy policy"
                    iconBackgroundColor={colors.accent[100]}
                    iconColor={colors.accent[600]}
                    onPress={() => { }}
                />

                <SettingItem
                    icon="information-circle"
                    title="About"
                    subtitle="App version and info"
                    iconBackgroundColor={colors.gray[100]}
                    iconColor={colors.gray[600]}
                    onPress={() => { }}
                />
            </ModernCard>

            {/* Footer spacing */}
            <View style={styles.footer} />
        </ScrollView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
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
    footer: {
        height: Spacing.xl,
    },
});
