import React, { useState, useRef } from 'react';
import { View, TextInput, Text, Animated, TouchableOpacity } from 'react-native';
import { useTheme } from '../theme';

interface ModernInputProps {
    label: string;
    value: string;
    onChangeText: (text: string) => void;
    placeholder?: string;
    secureTextEntry?: boolean;
    keyboardType?: 'default' | 'email-address' | 'numeric' | 'phone-pad';
    autoCapitalize?: 'none' | 'sentences' | 'words' | 'characters';
    error?: string;
    disabled?: boolean;
    multiline?: boolean;
    numberOfLines?: number;
    leftIcon?: React.ReactNode;
    rightIcon?: React.ReactNode;
    onRightIconPress?: () => void;
    className?: string;
}

export default function ModernInput({
    label,
    value,
    onChangeText,
    placeholder,
    secureTextEntry = false,
    keyboardType = 'default',
    autoCapitalize = 'sentences',
    error,
    disabled = false,
    multiline = false,
    numberOfLines = 1,
    leftIcon,
    rightIcon,
    onRightIconPress,
    className = '',
}: ModernInputProps) {
    const theme = useTheme();
    const [isFocused, setIsFocused] = useState(false);
    const animatedLabelPosition = useRef(new Animated.Value(value ? 1 : 0)).current;
    const animatedBorderColor = useRef(new Animated.Value(0)).current;

    const handleFocus = () => {
        setIsFocused(true);
        Animated.parallel([
            Animated.timing(animatedLabelPosition, {
                toValue: 1,
                duration: 200,
                useNativeDriver: false,
            }),
            Animated.timing(animatedBorderColor, {
                toValue: 1,
                duration: 200,
                useNativeDriver: false,
            }),
        ]).start();
    };

    const handleBlur = () => {
        setIsFocused(false);
        if (!value) {
            Animated.timing(animatedLabelPosition, {
                toValue: 0,
                duration: 200,
                useNativeDriver: false,
            }).start();
        }
        Animated.timing(animatedBorderColor, {
            toValue: 0,
            duration: 200,
            useNativeDriver: false,
        }).start();
    };

    const labelStyle = {
        position: 'absolute' as const,
        left: leftIcon ? 48 : 16,
        top: animatedLabelPosition.interpolate({
            inputRange: [0, 1],
            outputRange: [20, 4],
        }),
        fontSize: animatedLabelPosition.interpolate({
            inputRange: [0, 1],
            outputRange: [16, 12],
        }),
        color: animatedLabelPosition.interpolate({
            inputRange: [0, 1],
            outputRange: [theme.colors.onSurfaceVariant, theme.colors.primary],
        }),
        backgroundColor: theme.colors.surface,
        paddingHorizontal: 4,
        zIndex: 1,
    };

    const borderColor = animatedBorderColor.interpolate({
        inputRange: [0, 1],
        outputRange: [theme.colors.outline, error ? theme.colors.error : theme.colors.primary],
    });

    return (
        <View className={`mb-4 ${className}`}>
            <View className="relative">
                {/* Animated Label */}
                <Animated.Text style={labelStyle}>
                    {label}
                </Animated.Text>

                {/* Input Container */}
                <Animated.View
                    style={{
                        borderColor,
                        borderWidth: 2,
                        borderRadius: 12,
                        backgroundColor: theme.colors.surface,
                        minHeight: multiline ? 80 : 56,
                        shadowColor: theme.colors.shadow,
                        shadowOffset: {
                            width: 0,
                            height: 2,
                        },
                        shadowOpacity: isFocused ? 0.1 : 0.05,
                        shadowRadius: 8,
                        elevation: isFocused ? 2 : 1,
                    }}
                >
                    <View className={`flex-row items-${multiline ? 'start' : 'center'} px-4 py-3`}>
                        {/* Left Icon */}
                        {leftIcon && (
                            <View className="mr-3">
                                {leftIcon}
                            </View>
                        )}

                        {/* Text Input */}
                        <TextInput
                            value={value}
                            onChangeText={onChangeText}
                            placeholder={isFocused ? placeholder : ''}
                            placeholderTextColor={theme.colors.onSurfaceVariant}
                            secureTextEntry={secureTextEntry}
                            keyboardType={keyboardType}
                            autoCapitalize={autoCapitalize}
                            onFocus={handleFocus}
                            onBlur={handleBlur}
                            editable={!disabled}
                            multiline={multiline}
                            numberOfLines={numberOfLines}
                            style={{
                                flex: 1,
                                fontSize: 16,
                                color: theme.colors.onSurface,
                                paddingTop: multiline ? 20 : 16,
                                paddingBottom: 8,
                                textAlignVertical: multiline ? 'top' : 'center',
                                opacity: disabled ? 0.5 : 1,
                            }}
                        />

                        {/* Right Icon */}
                        {rightIcon && (
                            <TouchableOpacity
                                onPress={onRightIconPress}
                                disabled={!onRightIconPress}
                                className="ml-3"
                            >
                                {rightIcon}
                            </TouchableOpacity>
                        )}
                    </View>
                </Animated.View>
            </View>

            {/* Error Message */}
            {error && (
                <Animated.View
                    className="mt-2 px-2"
                    style={{
                        opacity: 1,
                    }}
                >
                    <Text
                        style={{
                            color: theme.colors.error,
                            fontSize: 12,
                            fontWeight: '400',
                        }}
                    >
                        {error}
                    </Text>
                </Animated.View>
            )}
        </View>
    );
}
