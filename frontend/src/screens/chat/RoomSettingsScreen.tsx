import React, { useState, useEffect } from 'react';
import { View, ScrollView, StyleSheet, Alert } from 'react-native';
import { Text, Card, Switch, Button, Divider, List } from 'react-native-paper';
import { StackScreenProps } from '@react-navigation/stack';
import { useChatStore } from '../../stores/chatStore';
import { useTheme, colors, Spacing } from '../../theme';
import { ChatStackParamList } from '../../navigation/MainNavigator';

type Props = StackScreenProps<ChatStackParamList, 'RoomSettings'>;

export default function RoomSettingsScreen({ route, navigation }: Props) {
  const { room } = route.params;
  const { leaveRoom } = useChatStore();
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [mentionNotifications, setMentionNotifications] = useState(true);
  const theme = useTheme();

  useEffect(() => {
    navigation.setOptions({
      title: `${room.name} Settings`,
    });
  }, [navigation, room]);

  const handleLeaveRoom = () => {
    Alert.alert(
      'Leave Room',
      `Are you sure you want to leave ${room.name}?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Leave',
          style: 'destructive',
          onPress: async () => {
            const success = await leaveRoom(room.id);
            if (success) {
              navigation.goBack();
            }
          },
        },
      ]
    );
  };

  const handleInviteMembers = () => {
    Alert.alert(
      'Invite Members',
      'This feature allows you to invite new members to the room.',
      [{ text: 'OK' }]
    );
  };

  const handleManageMembers = () => {
    Alert.alert(
      'Manage Members',
      'This feature allows you to manage room members and permissions.',
      [{ text: 'OK' }]
    );
  };

  const handleViewAnalytics = () => {
    Alert.alert(
      'Room Analytics',
      'This feature shows room activity and message statistics.',
      [{ text: 'OK' }]
    );
  };

  return (
    <ScrollView 
      style={[styles.container, { backgroundColor: theme.colors.background }]}
      contentContainerStyle={styles.content}
    >
      {/* Room Info */}
      <Card style={styles.card}>
        <Card.Content>
          <Text style={styles.sectionTitle}>Room Information</Text>
          <List.Item
            title="Room Name"
            description={room.name}
            left={(props) => <List.Icon {...props} icon="forum" />}
          />
          <List.Item
            title="Room Type"
            description={room.type === 'group' ? 'Group Chat' : 'Channel'}
            left={(props) => <List.Icon {...props} icon="account-group" />}
          />
          <List.Item
            title="Created"
            description={new Date(room.created_at).toLocaleDateString()}
            left={(props) => <List.Icon {...props} icon="calendar" />}
          />
          {room.description && (
            <List.Item
              title="Description"
              description={room.description}
              left={(props) => <List.Icon {...props} icon="text" />}
            />
          )}
        </Card.Content>
      </Card>

      {/* Notification Settings */}
      <Card style={styles.card}>
        <Card.Content>
          <Text style={styles.sectionTitle}>Notifications</Text>
          <View style={styles.settingRow}>
            <View style={styles.settingInfo}>
              <Text style={styles.settingTitle}>All Notifications</Text>
              <Text style={[styles.settingDescription, { color: theme.colors.onSurfaceVariant }]}>
                Receive notifications for all messages
              </Text>
            </View>
            <Switch
              value={notificationsEnabled}
              onValueChange={setNotificationsEnabled}
            />
          </View>
          
          <Divider style={styles.divider} />
          
          <View style={styles.settingRow}>
            <View style={styles.settingInfo}>
              <Text style={styles.settingTitle}>Mention Notifications</Text>
              <Text style={[styles.settingDescription, { color: theme.colors.onSurfaceVariant }]}>
                Only notify when mentioned
              </Text>
            </View>
            <Switch
              value={mentionNotifications}
              onValueChange={setMentionNotifications}
              disabled={!notificationsEnabled}
            />
          </View>
        </Card.Content>
      </Card>

      {/* Room Management */}
      <Card style={styles.card}>
        <Card.Content>
          <Text style={styles.sectionTitle}>Room Management</Text>
          
          <Button
            mode="outlined"
            onPress={() => navigation.navigate('EmojiManagement', { room })}
            icon="emoticon-happy"
            style={styles.actionButton}
            contentStyle={styles.buttonContent}
          >
            Manage Custom Emojis
          </Button>

          <Button
            mode="outlined"
            onPress={handleInviteMembers}
            icon="account-plus"
            style={styles.actionButton}
            contentStyle={styles.buttonContent}
          >
            Invite Members
          </Button>

          <Button
            mode="outlined"
            onPress={handleManageMembers}
            icon="account-group"
            style={styles.actionButton}
            contentStyle={styles.buttonContent}
          >
            Manage Members
          </Button>

          <Button
            mode="outlined"
            onPress={handleViewAnalytics}
            icon="chart-line"
            style={styles.actionButton}
            contentStyle={styles.buttonContent}
          >
            View Analytics
          </Button>
        </Card.Content>
      </Card>

      {/* Search */}
      <Card style={styles.card}>
        <Card.Content>
          <Text style={styles.sectionTitle}>Search & History</Text>
          
          <Button
            mode="outlined"
            onPress={() => navigation.navigate('MessageSearch', { room })}
            icon="magnify"
            style={styles.actionButton}
            contentStyle={styles.buttonContent}
          >
            Search Messages
          </Button>
        </Card.Content>
      </Card>

      {/* Danger Zone */}
      <Card style={[styles.card, styles.dangerCard]}>
        <Card.Content>
          <Text style={[styles.sectionTitle, { color: colors.error[600] }]}>
            Danger Zone
          </Text>
          
          <Button
            mode="contained"
            onPress={handleLeaveRoom}
            icon="exit-to-app"
            style={[styles.actionButton, { backgroundColor: colors.error[500] }]}
            contentStyle={styles.buttonContent}
            labelStyle={{ color: 'white' }}
          >
            Leave Room
          </Button>
        </Card.Content>
      </Card>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    padding: Spacing.md,
  },
  card: {
    marginBottom: Spacing.md,
  },
  dangerCard: {
    borderColor: colors.error[300],
    borderWidth: 1,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: Spacing.md,
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: Spacing.sm,
  },
  settingInfo: {
    flex: 1,
    marginRight: Spacing.md,
  },
  settingTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 2,
  },
  settingDescription: {
    fontSize: 14,
  },
  divider: {
    marginVertical: Spacing.sm,
  },
  actionButton: {
    marginBottom: Spacing.sm,
  },
  buttonContent: {
    justifyContent: 'flex-start',
    paddingVertical: Spacing.xs,
  },
});
