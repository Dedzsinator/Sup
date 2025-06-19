import React, { useState } from 'react';
import { View, StyleSheet, ScrollView, Alert, KeyboardAvoidingView, Platform } from 'react-native';
import { Text } from 'react-native-paper';
import { StackNavigationProp } from '@react-navigation/stack';
import { Ionicons } from '@expo/vector-icons';
import { useTheme, Spacing } from '../../theme';
import { useChatStore } from '../../stores/chatStore';
import { ChatStackParamList } from '../../navigation/MainNavigator';
import ModernCard from '../../components/ModernCard';
import ModernInput from '../../components/ModernInput';
import ModernButton from '../../components/ModernButton';

type CreateRoomScreenNavigationProp = StackNavigationProp<ChatStackParamList, 'CreateRoom'>;

interface Props {
    navigation: CreateRoomScreenNavigationProp;
}

export default function CreateRoomScreen({ navigation }: Props) {
    const [roomName, setRoomName] = useState('');
    const [description, setDescription] = useState('');
    const [roomType, setRoomType] = useState<'group' | 'channel'>('group');
    const [isPrivate, setIsPrivate] = useState(false);
    const [loading, setLoading] = useState(false);
    const { createRoom } = useChatStore();
    const theme = useTheme();

    const handleCreateRoom = async () => {
        if (!roomName.trim()) {
            Alert.alert('Error', 'Room name is required');
            return;
        }

        setLoading(true);
        try {
            const success = await createRoom(roomName.trim(), roomType, description.trim() || undefined);
            if (success) {
                Alert.alert('Success', 'Room created successfully', [
                    { text: 'OK', onPress: () => navigation.goBack() }
                ]);
            } else {
                Alert.alert('Error', 'Failed to create room');
            }
        } catch {
            Alert.alert('Error', 'Failed to create room');
        } finally {
            setLoading(false);
        }
    };

    return (
        <KeyboardAvoidingView
            style={[styles.container, { backgroundColor: theme.colors.background }]}
            behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        >
            <ScrollView
                contentContainerStyle={styles.scrollContent}
                showsVerticalScrollIndicator={false}
            >
                {/* Header */}
                <View style={styles.header}>
                    <Text style={[styles.headerTitle, { color: theme.colors.onBackground }]}>
                        Create New Room
                    </Text>
                    <Text style={[styles.headerSubtitle, { color: theme.colors.onSurfaceVariant }]}>
                        Start a new conversation or community
                    </Text>
                </View>

                {/* Form */}
                <ModernCard variant="elevated" padding="lg" borderRadius="xl" style={styles.formCard}>
                    <ModernInput
                        label="Room Name"
                        value={roomName}
                        onChangeText={setRoomName}
                        placeholder="Enter room name"
                        leftIcon={
                            <Ionicons
                                name="chatbubbles-outline"
                                size={20}
                                color={theme.colors.onSurfaceVariant}
                            />
                        }
                    />

                    <ModernInput
                        label="Description (Optional)"
                        value={description}
                        onChangeText={setDescription}
                        placeholder="Describe what this room is about"
                        multiline
                        numberOfLines={3}
                        leftIcon={
                            <Ionicons
                                name="information-outline"
                                size={20}
                                color={theme.colors.onSurfaceVariant}
                            />
                        }
                    />

                    {/* Room Type Selection */}
                    <View style={styles.sectionContainer}>
                        <Text style={[styles.sectionTitle, { color: theme.colors.onSurface }]}>
                            Room Type
                        </Text>

                        <View style={styles.optionRow}>
                            <ModernButton
                                title="Group Chat"
                                onPress={() => setRoomType('group')}
                                variant={roomType === 'group' ? 'primary' : 'outline'}
                                size="md"
                                icon={<Ionicons name="people" size={16} color={roomType === 'group' ? 'white' : theme.colors.primary} />}
                            />
                            <View style={styles.optionSpacer} />
                            <ModernButton
                                title="Channel"
                                onPress={() => setRoomType('channel')}
                                variant={roomType === 'channel' ? 'primary' : 'outline'}
                                size="md"
                                icon={<Ionicons name="megaphone" size={16} color={roomType === 'channel' ? 'white' : theme.colors.primary} />}
                            />
                        </View>

                        <Text style={[styles.optionDescription, { color: theme.colors.onSurfaceVariant }]}>
                            {roomType === 'group'
                                ? 'Group chats are perfect for small teams and friends'
                                : 'Channels are great for announcements and larger communities'
                            }
                        </Text>
                    </View>

                    {/* Privacy Toggle */}
                    <View style={styles.sectionContainer}>
                        <View style={styles.privacyRow}>
                            <View style={styles.privacyInfo}>
                                <Text style={[styles.sectionTitle, { color: theme.colors.onSurface }]}>
                                    Private Room
                                </Text>
                                <Text style={[styles.privacyDescription, { color: theme.colors.onSurfaceVariant }]}>
                                    Only invited members can join
                                </Text>
                            </View>
                            <ModernButton
                                title={isPrivate ? 'ON' : 'OFF'}
                                onPress={() => setIsPrivate(!isPrivate)}
                                variant={isPrivate ? 'primary' : 'outline'}
                                size="sm"
                            />
                        </View>
                    </View>

                    {/* Action Buttons */}
                    <View style={styles.actionsContainer}>
                        <ModernButton
                            title="Cancel"
                            onPress={() => navigation.goBack()}
                            variant="outline"
                            size="lg"
                            disabled={loading}
                        />
                        <View style={styles.actionSpacer} />
                        <ModernButton
                            title="Create Room"
                            onPress={handleCreateRoom}
                            variant="primary"
                            size="lg"
                            loading={loading}
                            disabled={loading || !roomName.trim()}
                        />
                    </View>
                </ModernCard>
            </ScrollView>
        </KeyboardAvoidingView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    scrollContent: {
        paddingHorizontal: Spacing.lg,
        paddingVertical: Spacing.md,
    },
    header: {
        marginBottom: Spacing.lg,
    },
    headerTitle: {
        fontSize: 28,
        fontWeight: 'bold',
        marginBottom: Spacing.xs,
    },
    headerSubtitle: {
        fontSize: 16,
        lineHeight: 22,
    },
    formCard: {
        marginBottom: Spacing.lg,
    },
    sectionContainer: {
        marginTop: Spacing.lg,
    },
    sectionTitle: {
        fontSize: 16,
        fontWeight: '600',
        marginBottom: Spacing.sm,
    },
    optionRow: {
        flexDirection: 'row',
        marginBottom: Spacing.sm,
    },
    optionSpacer: {
        width: Spacing.sm,
    },
    optionDescription: {
        fontSize: 14,
        lineHeight: 18,
    },
    privacyRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    privacyInfo: {
        flex: 1,
        marginRight: Spacing.md,
    },
    privacyDescription: {
        fontSize: 14,
        lineHeight: 18,
        marginTop: 2,
    },
    actionsContainer: {
        flexDirection: 'row',
        marginTop: Spacing.xl,
    },
    actionSpacer: {
        width: Spacing.md,
    },
});
