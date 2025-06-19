import React, { useState } from 'react';
import { View, StyleSheet, ScrollView, Alert, TouchableOpacity } from 'react-native';
import { Text, Avatar } from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import { useAuthStore } from '../../stores/authStore';
import { useTheme, colors, Spacing } from '../../theme';
import ModernCard from '../../components/ModernCard';
import ModernButton from '../../components/ModernButton';
import ModernInput from '../../components/ModernInput';

export default function ProfileScreen() {
    const { user, updateProfile, logout } = useAuthStore();
    const [editing, setEditing] = useState(false);
    const [username, setUsername] = useState(user?.username || '');
    const [loading, setLoading] = useState(false);
    const theme = useTheme();

    const handleSave = async () => {
        if (!username.trim()) {
            Alert.alert('Error', 'Username cannot be empty');
            return;
        }

        setLoading(true);
        const success = await updateProfile({ username: username.trim() });
        setLoading(false);

        if (success) {
            setEditing(false);
            Alert.alert('Success', 'Profile updated successfully');
        } else {
            Alert.alert('Error', 'Failed to update profile');
        }
    };

    const handleCancel = () => {
        setUsername(user?.username || '');
        setEditing(false);
    };

    const handleLogout = () => {
        Alert.alert(
            'Logout',
            'Are you sure you want to logout?',
            [
                { text: 'Cancel', style: 'cancel' },
                { text: 'Logout', style: 'destructive', onPress: logout },
            ]
        );
    };

    if (!user) return null;

    return (
        <ScrollView
            style={[styles.container, { backgroundColor: theme.colors.background }]}
            contentContainerStyle={styles.scrollContent}
            showsVerticalScrollIndicator={false}
        >
            {/* Header */}
            <View style={styles.header}>
                <Text style={[styles.headerTitle, { color: theme.colors.onBackground }]}>
                    Profile
                </Text>
            </View>

            {/* Profile Card */}
            <ModernCard variant="elevated" padding="lg" borderRadius="lg" style={styles.profileCard}>
                {/* Avatar Section */}
                <View style={styles.avatarSection}>
                    <View style={[styles.avatarContainer, { backgroundColor: colors.primary[100] }]}>
                        <Avatar.Text
                            size={80}
                            label={user.username.substring(0, 2).toUpperCase()}
                            style={[styles.avatar, { backgroundColor: colors.primary[500] }]}
                            labelStyle={{ fontSize: 24, fontWeight: 'bold', color: 'white' }}
                        />

                        {/* Online status indicator */}
                        <View style={[
                            styles.statusIndicator,
                            { backgroundColor: user.is_online ? colors.accent[500] : colors.gray[400] }
                        ]} />
                    </View>

                    {/* User Status */}
                    <View style={styles.statusContainer}>
                        <View style={styles.statusRow}>
                            <Ionicons
                                name={user.is_online ? 'radio-button-on' : 'radio-button-off'}
                                size={16}
                                color={user.is_online ? colors.accent[500] : colors.gray[400]}
                            />
                            <Text style={[
                                styles.statusText,
                                { color: user.is_online ? colors.accent[600] : colors.gray[500] }
                            ]}>
                                {user.is_online ? 'Online' : 'Offline'}
                            </Text>
                        </View>
                    </View>
                </View>

                {/* Profile Info */}
                <View style={styles.profileInfo}>
                    {editing ? (
                        <View style={styles.editForm}>
                            <ModernInput
                                label="Username"
                                value={username}
                                onChangeText={setUsername}
                                placeholder="Enter your username"
                                leftIcon={
                                    <Ionicons
                                        name="person-outline"
                                        size={20}
                                        color={theme.colors.onSurfaceVariant}
                                    />
                                }
                            />

                            <View style={styles.editActions}>
                                <ModernButton
                                    title="Cancel"
                                    onPress={handleCancel}
                                    variant="ghost"
                                    size="md"
                                />
                                <ModernButton
                                    title="Save"
                                    onPress={handleSave}
                                    loading={loading}
                                    disabled={loading}
                                    size="md"
                                />
                            </View>
                        </View>
                    ) : (
                        <View style={styles.displayInfo}>
                            <View style={styles.infoItem}>
                                <Text style={[styles.infoLabel, { color: theme.colors.onSurfaceVariant }]}>
                                    Username
                                </Text>
                                <Text style={[styles.infoValue, { color: theme.colors.onSurface }]}>
                                    {user.username}
                                </Text>
                            </View>

                            <View style={styles.infoItem}>
                                <Text style={[styles.infoLabel, { color: theme.colors.onSurfaceVariant }]}>
                                    Email
                                </Text>
                                <Text style={[styles.infoValue, { color: theme.colors.onSurface }]}>
                                    {user.email}
                                </Text>
                            </View>

                            <ModernButton
                                title="Edit Profile"
                                onPress={() => setEditing(true)}
                                variant="outline"
                                size="md"
                            />
                        </View>
                    )}
                </View>
            </ModernCard>

            {/* Stats Card */}
            <ModernCard variant="elevated" padding="lg" borderRadius="xl" style={styles.statsCard}>
                <Text style={[styles.sectionTitle, { color: theme.colors.onSurface }]}>
                    Activity
                </Text>

                <View style={styles.statsGrid}>
                    <View style={styles.statItem}>
                        <View style={[styles.statIconContainer, { backgroundColor: colors.primary[100] }]}>
                            <Ionicons name="chatbubbles" size={20} color={colors.primary[600]} />
                        </View>
                        <Text style={[styles.statValue, { color: theme.colors.onSurface }]}>
                            12
                        </Text>
                        <Text style={[styles.statLabel, { color: theme.colors.onSurfaceVariant }]}>
                            Chats
                        </Text>
                    </View>

                    <View style={styles.statItem}>
                        <View style={[styles.statIconContainer, { backgroundColor: colors.secondary[100] }]}>
                            <Ionicons name="send" size={20} color={colors.secondary[600]} />
                        </View>
                        <Text style={[styles.statValue, { color: theme.colors.onSurface }]}>
                            148
                        </Text>
                        <Text style={[styles.statLabel, { color: theme.colors.onSurfaceVariant }]}>
                            Messages
                        </Text>
                    </View>

                    <View style={styles.statItem}>
                        <View style={[styles.statIconContainer, { backgroundColor: colors.accent[100] }]}>
                            <Ionicons name="people" size={20} color={colors.accent[600]} />
                        </View>
                        <Text style={[styles.statValue, { color: theme.colors.onSurface }]}>
                            8
                        </Text>
                        <Text style={[styles.statLabel, { color: theme.colors.onSurfaceVariant }]}>
                            Groups
                        </Text>
                    </View>
                </View>
            </ModernCard>

            {/* Settings Card */}
            <ModernCard variant="elevated" padding="lg" borderRadius="xl" style={styles.settingsCard}>
                <Text style={[styles.sectionTitle, { color: theme.colors.onSurface }]}>
                    Settings
                </Text>

                <TouchableOpacity style={styles.settingItem} activeOpacity={0.7}>
                    <View style={[styles.settingIconContainer, { backgroundColor: colors.primary[100] }]}>
                        <Ionicons name="notifications" size={20} color={colors.primary[600]} />
                    </View>
                    <View style={styles.settingContent}>
                        <Text style={[styles.settingTitle, { color: theme.colors.onSurface }]}>
                            Notifications
                        </Text>
                        <Text style={[styles.settingSubtitle, { color: theme.colors.onSurfaceVariant }]}>
                            Manage notification preferences
                        </Text>
                    </View>
                    <Ionicons name="chevron-forward" size={20} color={theme.colors.onSurfaceVariant} />
                </TouchableOpacity>

                <TouchableOpacity style={styles.settingItem} activeOpacity={0.7}>
                    <View style={[styles.settingIconContainer, { backgroundColor: colors.secondary[100] }]}>
                        <Ionicons name="shield-checkmark" size={20} color={colors.secondary[600]} />
                    </View>
                    <View style={styles.settingContent}>
                        <Text style={[styles.settingTitle, { color: theme.colors.onSurface }]}>
                            Privacy & Security
                        </Text>
                        <Text style={[styles.settingSubtitle, { color: theme.colors.onSurfaceVariant }]}>
                            Control your privacy settings
                        </Text>
                    </View>
                    <Ionicons name="chevron-forward" size={20} color={theme.colors.onSurfaceVariant} />
                </TouchableOpacity>

                <TouchableOpacity style={styles.settingItem} activeOpacity={0.7}>
                    <View style={[styles.settingIconContainer, { backgroundColor: colors.accent[100] }]}>
                        <Ionicons name="color-palette" size={20} color={colors.accent[600]} />
                    </View>
                    <View style={styles.settingContent}>
                        <Text style={[styles.settingTitle, { color: theme.colors.onSurface }]}>
                            Appearance
                        </Text>
                        <Text style={[styles.settingSubtitle, { color: theme.colors.onSurfaceVariant }]}>
                            Customize theme and colors
                        </Text>
                    </View>
                    <Ionicons name="chevron-forward" size={20} color={theme.colors.onSurfaceVariant} />
                </TouchableOpacity>
            </ModernCard>

            {/* Logout Button */}
            <TouchableOpacity
                style={[styles.logoutButton, { backgroundColor: colors.error[500] }]}
                onPress={handleLogout}
                activeOpacity={0.8}
            >
                <Ionicons name="log-out-outline" size={20} color="white" />
                <Text style={styles.logoutButtonText}>
                    Logout
                </Text>
            </TouchableOpacity>

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
    profileCard: {
        marginBottom: Spacing.lg,
    },
    avatarSection: {
        alignItems: 'center',
        marginBottom: Spacing.xl,
    },
    avatarContainer: {
        position: 'relative',
        padding: 12,
        borderRadius: 56,
    },
    avatar: {
        elevation: 4,
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 4,
    },
    statusIndicator: {
        position: 'absolute',
        bottom: 8,
        right: 8,
        width: 16,
        height: 16,
        borderRadius: 8,
        borderWidth: 3,
        borderColor: 'white',
    },
    statusContainer: {
        marginTop: Spacing.md,
    },
    statusRow: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
    },
    statusText: {
        marginLeft: Spacing.xs,
        fontSize: 14,
        fontWeight: '500',
    },
    profileInfo: {
        width: '100%',
    },
    editForm: {
        gap: Spacing.lg,
    },
    editActions: {
        flexDirection: 'row',
        marginTop: Spacing.sm,
        gap: Spacing.sm,
    },
    displayInfo: {
        gap: Spacing.lg,
    },
    infoItem: {
        gap: Spacing.xs,
    },
    infoLabel: {
        fontSize: 14,
        fontWeight: '500',
    },
    infoValue: {
        fontSize: 16,
        fontWeight: '400',
    },
    statsCard: {
        marginBottom: Spacing.lg,
    },
    sectionTitle: {
        fontSize: 18,
        fontWeight: '600',
        marginBottom: Spacing.md,
    },
    statsGrid: {
        flexDirection: 'row',
        justifyContent: 'space-around',
    },
    statItem: {
        alignItems: 'center',
        flex: 1,
    },
    statIconContainer: {
        width: 40,
        height: 40,
        borderRadius: 20,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: Spacing.sm,
    },
    statValue: {
        fontSize: 20,
        fontWeight: 'bold',
        marginBottom: 2,
    },
    statLabel: {
        fontSize: 12,
        textAlign: 'center',
    },
    settingsCard: {
        marginBottom: Spacing.lg,
    },
    settingItem: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingVertical: Spacing.md,
        borderBottomWidth: 1,
        borderBottomColor: 'rgba(0,0,0,0.05)',
    },
    settingIconContainer: {
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
    logoutButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: Spacing.md,
        paddingHorizontal: Spacing.lg,
        borderRadius: 12,
        marginBottom: Spacing.lg,
        gap: Spacing.sm,
    },
    logoutButtonText: {
        color: 'white',
        fontSize: 16,
        fontWeight: '600',
    },
    footer: {
        height: Spacing.xl,
    },
});
