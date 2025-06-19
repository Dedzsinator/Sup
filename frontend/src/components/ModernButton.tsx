import React from 'react';
import { TouchableOpacity, Text, View, ViewStyle, TextStyle } from 'react-native';
import { useTheme } from '../theme';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withSpring,
    withTiming,
    runOnJS
} from 'react-native-reanimated';

const AnimatedTouchableOpacity = Animated.createAnimatedComponent(TouchableOpacity);

interface ModernButtonProps {
    title: string;
    onPress: () => void;
    variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
    size?: 'sm' | 'md' | 'lg';
    disabled?: boolean;
    loading?: boolean;
    icon?: React.ReactNode;
    fullWidth?: boolean;
    className?: string;
}

export default function ModernButton({
    title,
    onPress,
    variant = 'primary',
    size = 'md',
    disabled = false,
    loading = false,
    icon,
    fullWidth = false,
    className = '',
}: ModernButtonProps) {
    const theme = useTheme();
    const scale = useSharedValue(1);
    const opacity = useSharedValue(1);

    const handlePressIn = () => {
        scale.value = withSpring(0.95);
        opacity.value = withTiming(0.8);
    };

    const handlePressOut = () => {
        scale.value = withSpring(1);
        opacity.value = withTiming(1);
        if (!disabled && !loading) {
            runOnJS(onPress)();
        }
    };

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: scale.value }],
        opacity: opacity.value,
    }));

    const getVariantStyles = (): { container: ViewStyle; text: TextStyle } => {
        switch (variant) {
            case 'primary':
                return {
                    container: {
                        backgroundColor: theme.colors.primary,
                        borderWidth: 0,
                    },
                    text: {
                        color: theme.colors.onPrimary,
                    },
                };
            case 'secondary':
                return {
                    container: {
                        backgroundColor: theme.colors.secondary,
                        borderWidth: 0,
                    },
                    text: {
                        color: theme.colors.onSecondary,
                    },
                };
            case 'outline':
                return {
                    container: {
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        borderColor: theme.colors.primary,
                    },
                    text: {
                        color: theme.colors.primary,
                    },
                };
            case 'ghost':
                return {
                    container: {
                        backgroundColor: 'transparent',
                        borderWidth: 0,
                    },
                    text: {
                        color: theme.colors.primary,
                    },
                };
            default:
                return {
                    container: {
                        backgroundColor: theme.colors.primary,
                        borderWidth: 0,
                    },
                    text: {
                        color: theme.colors.onPrimary,
                    },
                };
        }
    };

    const getSizeStyles = (): { container: ViewStyle; text: TextStyle } => {
        switch (size) {
            case 'sm':
                return {
                    container: {
                        paddingHorizontal: 16,
                        paddingVertical: 8,
                        borderRadius: 8,
                    },
                    text: {
                        fontSize: 14,
                        fontWeight: '500',
                    },
                };
            case 'md':
                return {
                    container: {
                        paddingHorizontal: 24,
                        paddingVertical: 12,
                        borderRadius: 12,
                    },
                    text: {
                        fontSize: 16,
                        fontWeight: '600',
                    },
                };
            case 'lg':
                return {
                    container: {
                        paddingHorizontal: 32,
                        paddingVertical: 16,
                        borderRadius: 16,
                    },
                    text: {
                        fontSize: 18,
                        fontWeight: '600',
                    },
                };
            default:
                return {
                    container: {
                        paddingHorizontal: 24,
                        paddingVertical: 12,
                        borderRadius: 12,
                    },
                    text: {
                        fontSize: 16,
                        fontWeight: '600',
                    },
                };
        }
    };

    const variantStyles = getVariantStyles();
    const sizeStyles = getSizeStyles();

    return (
        <AnimatedTouchableOpacity
            onPressIn={handlePressIn}
            onPressOut={handlePressOut}
            disabled={disabled || loading}
            style={[
                animatedStyle,
                {
                    flexDirection: 'row',
                    alignItems: 'center',
                    justifyContent: 'center',
                    shadowColor: theme.colors.shadow,
                    shadowOffset: {
                        width: 0,
                        height: 4,
                    },
                    shadowOpacity: variant === 'primary' || variant === 'secondary' ? 0.2 : 0,
                    shadowRadius: 8,
                    elevation: variant === 'primary' || variant === 'secondary' ? 4 : 0,
                    opacity: disabled ? 0.5 : 1,
                    width: fullWidth ? '100%' : undefined,
                },
                variantStyles.container,
                sizeStyles.container,
            ]}
            className={className}
        >
            {loading ? (
                <View className="mr-2">
                    <Animated.View
                        className="w-4 h-4 border-2 border-white rounded-full"
                        style={{
                            borderTopColor: 'transparent',
                        }}
                    />
                </View>
            ) : icon ? (
                <View className="mr-2">
                    {icon}
                </View>
            ) : null}

            <Text
                style={[
                    variantStyles.text,
                    sizeStyles.text,
                    {
                        textAlign: 'center',
                    },
                ]}
            >
                {title}
            </Text>
        </AnimatedTouchableOpacity>
    );
}
