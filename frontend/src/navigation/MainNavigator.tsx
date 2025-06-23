import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../theme';

// Screens
import ChatListScreen from '../screens/chat/ChatListScreen';
import ChatScreen from '../screens/chat/ChatScreen';
import CreateRoomScreen from '../screens/chat/CreateRoomScreen';
import ThreadScreen from '../screens/chat/ThreadScreen';
import MessageSearchScreen from '../screens/chat/MessageSearchScreen';
import EmojiManagementScreen from '../screens/chat/EmojiManagementScreen';
import RoomSettingsScreen from '../screens/chat/RoomSettingsScreen';
import ProfileScreen from '../screens/profile/ProfileScreen';
import SettingsScreen from '../screens/settings/SettingsScreen';

// Types
import { Room, MessageThread } from '../types';

export type MainTabParamList = {
    ChatsTab: undefined;
    Profile: undefined;
    Settings: undefined;
};

export type ChatStackParamList = {
    ChatList: undefined;
    Chat: { room: Room };
    CreateRoom: undefined;
    ThreadView: { thread: MessageThread };
    MessageSearch: { room?: Room };
    EmojiManagement: { room: Room };
    RoomSettings: { room: Room };
};

const Tab = createBottomTabNavigator<MainTabParamList>();
const ChatStack = createStackNavigator<ChatStackParamList>();

function ChatNavigator() {
    const theme = useTheme();

    return (
        <ChatStack.Navigator
            screenOptions={{
                headerStyle: {
                    backgroundColor: theme.colors.surface,
                    elevation: 0,
                    shadowOpacity: 0,
                    borderBottomWidth: 1,
                    borderBottomColor: 'rgba(0,0,0,0.1)',
                },
                headerTitleStyle: {
                    fontWeight: '600',
                    fontSize: 18,
                    color: theme.colors.onSurface,
                },
                headerBackTitleVisible: false,
                headerTintColor: theme.colors.primary,
            }}
        >
            <ChatStack.Screen
                name="ChatList"
                component={ChatListScreen}
                options={{
                    title: 'Chats',
                    headerShown: false
                }}
            />
            <ChatStack.Screen
                name="Chat"
                component={ChatScreen}
                options={({ route }) => ({
                    title: route.params.room.name,
                    headerBackTitle: 'Back'
                })}
            />
            <ChatStack.Screen
                name="CreateRoom"
                component={CreateRoomScreen}
                options={{
                    title: 'Create Room',
                    headerBackTitle: 'Back'
                }}
            />
            <ChatStack.Screen
                name="ThreadView"
                component={ThreadScreen}
                options={({ route }) => ({
                    title: `Thread`,
                    headerBackTitle: 'Back'
                })}
            />
            <ChatStack.Screen
                name="MessageSearch"
                component={MessageSearchScreen}
                options={{
                    title: 'Search Messages',
                    headerBackTitle: 'Back'
                }}
            />
            <ChatStack.Screen
                name="EmojiManagement"
                component={EmojiManagementScreen}
                options={{
                    title: 'Custom Emojis',
                    headerBackTitle: 'Back'
                }}
            />
            <ChatStack.Screen
                name="RoomSettings"
                component={RoomSettingsScreen}
                options={{
                    title: 'Room Settings',
                    headerBackTitle: 'Back'
                }}
            />
        </ChatStack.Navigator>
    );
}

export default function MainNavigator() {
    const theme = useTheme();

    return (
        <Tab.Navigator
            screenOptions={({ route }) => ({
                tabBarIcon: ({ focused, color, size }) => {
                    let iconName: keyof typeof Ionicons.glyphMap;

                    if (route.name === 'ChatsTab') {
                        iconName = focused ? 'chatbubbles' : 'chatbubbles-outline';
                    } else if (route.name === 'Profile') {
                        iconName = focused ? 'person' : 'person-outline';
                    } else if (route.name === 'Settings') {
                        iconName = focused ? 'settings' : 'settings-outline';
                    } else {
                        iconName = 'help-outline';
                    }

                    return <Ionicons name={iconName} size={size} color={color} />;
                },
                tabBarActiveTintColor: theme.colors.primary,
                tabBarInactiveTintColor: theme.colors.onSurfaceVariant,
                tabBarStyle: {
                    backgroundColor: theme.colors.surface,
                    borderTopWidth: 1,
                    borderTopColor: 'rgba(0,0,0,0.1)',
                    paddingBottom: 8,
                    paddingTop: 8,
                    height: 65,
                },
                tabBarLabelStyle: {
                    fontSize: 12,
                    fontWeight: '500',
                    marginTop: 4,
                },
                headerShown: false,
                tabBarItemStyle: {
                    paddingVertical: 4,
                },
            })}
        >
            <Tab.Screen
                name="ChatsTab"
                component={ChatNavigator}
                options={{
                    title: 'Chats',
                    tabBarLabel: 'Chats'
                }}
            />
            <Tab.Screen
                name="Profile"
                component={ProfileScreen}
                options={{
                    tabBarLabel: 'Profile'
                }}
            />
            <Tab.Screen
                name="Settings"
                component={SettingsScreen}
                options={{
                    tabBarLabel: 'Settings'
                }}
            />
        </Tab.Navigator>
    );
}
