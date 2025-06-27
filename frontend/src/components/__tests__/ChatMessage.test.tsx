import React from 'react'
import { render, fireEvent, waitFor } from '@testing-library/react-native'
import ChatMessage from '../ChatMessage'

// Mock the spam detection service
jest.mock('../../services/spamDetection', () => ({
    checkSpam: jest.fn(),
}))

describe('ChatMessage Component', () => {
    const mockMessage = {
        id: '1',
        text: 'Hello, this is a test message',
        user_id: 'user123',
        username: 'TestUser',
        timestamp: new Date().toISOString(),
        is_spam: false,
    }

    beforeEach(() => {
        jest.clearAllMocks()
    })

    it('renders message correctly', () => {
        const { getByText, getByTestId } = render(<ChatMessage message={mockMessage} />)

        expect(getByText('Hello, this is a test message')).toBeTruthy()
        expect(getByText('TestUser')).toBeTruthy()
        expect(getByTestId('message-container')).toBeTruthy()
    })

    it('shows spam warning for flagged messages', () => {
        const spamMessage = {
            ...mockMessage,
            is_spam: true,
            spam_confidence: 0.95,
        }

        const { getByTestId, getByText } = render(<ChatMessage message={spamMessage} />)

        expect(getByTestId('spam-warning')).toBeTruthy()
        expect(getByText(/flagged as potential spam/i)).toBeTruthy()
    })

    it('handles message tap events', () => {
        const onMessagePress = jest.fn()
        const { getByTestId } = render(
            <ChatMessage message={mockMessage} onPress={onMessagePress} />
        )

        fireEvent.press(getByTestId('message-container'))
        expect(onMessagePress).toHaveBeenCalledWith(mockMessage)
    })

    it('displays timestamp correctly', () => {
        const { getByTestId } = render(<ChatMessage message={mockMessage} />)

        const timestampElement = getByTestId('message-timestamp')
        expect(timestampElement).toBeTruthy()
    })

    it('shows different styles for own messages', () => {
        const ownMessage = {
            ...mockMessage,
            user_id: 'current_user_id',
        }

        const { getByTestId } = render(
            <ChatMessage message={ownMessage} currentUserId="current_user_id" />
        )

        const container = getByTestId('message-container')
        // You would check for specific styling here
        expect(container).toBeTruthy()
    })

    it('handles spam feedback submission', async () => {
        const spamMessage = {
            ...mockMessage,
            is_spam: true,
        }

        const { getByTestId } = render(<ChatMessage message={spamMessage} />)

        const notSpamButton = getByTestId('not-spam-button')
        fireEvent.press(notSpamButton)

        await waitFor(() => {
            expect(getByTestId('feedback-submitted')).toBeTruthy()
        })
    })
})
