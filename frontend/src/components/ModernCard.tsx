import React from 'react';
import { View, ViewStyle } from 'react-native';
import { useTheme } from '../theme';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withSpring,
    interpolate,
    Extrapolate,
} from 'react-native-reanimated';

interface ModernCardProps {
    children: React.ReactNode;
    variant?: 'default' | 'elevated' | 'outlined' | 'glass';
    padding?: 'none' | 'sm' | 'md' | 'lg';
    margin?: 'none' | 'sm' | 'md' | 'lg';
    borderRadius?: 'sm' | 'md' | 'lg' | 'xl';
    pressable?: boolean;
    onPress?: () => void;
    className?: string;
    style?: ViewStyle;
}

const AnimatedView = Animated.createAnimatedComponent(View);

export default function ModernCard({
    children,
    variant = 'default',
    padding = 'md',
    margin = 'none',
    borderRadius = 'lg',
    pressable = false,
    onPress,
    className = '',
    style,
}: ModernCardProps) {
    const theme = useTheme();
    const scale = useSharedValue(1);
    const opacity = useSharedValue(1);

    const handlePressIn = () => {
        if (pressable) {
            scale.value = withSpring(0.98);
            opacity.value = withSpring(0.9);
        }
    };

    const handlePressOut = () => {
        if (pressable) {
            scale.value = withSpring(1);
            opacity.value = withSpring(1);
            onPress?.();
        }
    };

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: scale.value }],
        opacity: opacity.value,
    }));

    const getVariantStyles = (): ViewStyle => {
        switch (variant) {
            case 'elevated':
                return {
                    backgroundColor: theme.colors.surface,
                    shadowColor: theme.colors.shadow,
                    shadowOffset: {
                        width: 0,
                        height: 4,
                    },
                    shadowOpacity: 0.12,
                    shadowRadius: 12,
                    elevation: 8,
                };
            case 'outlined':
                return {
                    backgroundColor: theme.colors.surface,
                    borderWidth: 1,
                    borderColor: theme.colors.outline,
                };
            case 'glass':
                return {
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    shadowColor: theme.colors.shadow,
                    shadowOffset: {
                        width: 0,
                        height: 8,
                    },
                    shadowOpacity: 0.1,
                    shadowRadius: 16,
                    elevation: 4,
                };
            default:
                return {
                    backgroundColor: theme.colors.surface,
                    shadowColor: theme.colors.shadow,
                    shadowOffset: {
                        width: 0,
                        height: 2,
                    },
                    shadowOpacity: 0.08,
                    shadowRadius: 8,
                    elevation: 2,
                };
        }
    };

    const getPaddingStyles = (): ViewStyle => {
        switch (padding) {
            case 'none':
                return {};
            case 'sm':
                return { padding: 12 };
            case 'md':
                return { padding: 16 };
            case 'lg':
                return { padding: 24 };
            default:
                return { padding: 16 };
        }
    };

    const getMarginStyles = (): ViewStyle => {
        switch (margin) {
            case 'none':
                return {};
            case 'sm':
                return { margin: 8 };
            case 'md':
                return { margin: 16 };
            case 'lg':
                return { margin: 24 };
            default:
                return {};
        }
    };

    const getBorderRadiusStyles = (): ViewStyle => {
        switch (borderRadius) {
            case 'sm':
                return { borderRadius: 8 };
            case 'md':
                return { borderRadius: 12 };
            case 'lg':
                return { borderRadius: 16 };
            case 'xl':
                return { borderRadius: 24 };
            default:
                return { borderRadius: 16 };
        }
    };

    const combinedStyles: ViewStyle = {
        ...getVariantStyles(),
        ...getPaddingStyles(),
        ...getMarginStyles(),
        ...getBorderRadiusStyles(),
        ...style,
    };

    if (pressable) {
        return (
            <AnimatedView
                style={[animatedStyle, combinedStyles]}
                className={className}
                onTouchStart={handlePressIn}
                onTouchEnd={handlePressOut}
                onTouchCancel={handlePressOut}
            >
                {children}
            </AnimatedView>
        );
    }

    return (
        <View style={combinedStyles} className={className}>
            {children}
        </View>
    );
}
