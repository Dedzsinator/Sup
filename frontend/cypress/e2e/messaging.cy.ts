describe('Messaging Flow', () => {
  beforeEach(() => {
    // Login before each test
    cy.login('test@example.com', 'password123')
  })

  it('should create a new chat room', () => {
    cy.createChat('Test Chat Room')
    cy.contains('Test Chat Room').should('be.visible')
  })

  it('should send and receive messages', () => {
    cy.createChat('Message Test Room')
    cy.contains('Message Test Room').click()
    
    const testMessage = 'Hello, this is a test message!'
    cy.sendMessage(testMessage)
    cy.waitForMessage(testMessage)
  })

  it('should detect spam messages', () => {
    cy.createChat('Spam Test Room')
    cy.contains('Spam Test Room').click()
    
    const spamMessage = 'BUY CHEAP VIAGRA NOW! CLICK HERE FOR FREE MONEY!'
    cy.sendMessage(spamMessage)
    
    // Should show spam warning or filter the message
    cy.get('[data-testid="spam-warning"]').should('be.visible')
      .and('contain', 'Message flagged as potential spam')
  })

  it('should show message history', () => {
    cy.createChat('History Test Room')
    cy.contains('History Test Room').click()
    
    // Send multiple messages
    cy.sendMessage('First message')
    cy.sendMessage('Second message')
    cy.sendMessage('Third message')
    
    // All messages should be visible
    cy.waitForMessage('First message')
    cy.waitForMessage('Second message')
    cy.waitForMessage('Third message')
    
    // Check message order
    cy.get('[data-testid="message"]').should('have.length', 3)
  })

  it('should handle real-time message updates', () => {
    cy.createChat('Real-time Test Room')
    cy.contains('Real-time Test Room').click()
    
    // Simulate another user sending a message (this would normally come from WebSocket)
    cy.window().then((win) => {
      // Trigger a mock WebSocket message
      const mockMessage = {
        id: 'mock-message-id',
        text: 'Message from another user',
        user_id: 'other-user',
        timestamp: new Date().toISOString()
      }
      
      // This would depend on your WebSocket implementation
      win.dispatchEvent(new CustomEvent('mockMessage', { detail: mockMessage }))
    })
    
    cy.waitForMessage('Message from another user')
  })

  it('should handle message encryption/decryption', () => {
    cy.createChat('Encrypted Test Room')
    cy.contains('Encrypted Test Room').click()
    
    const sensitiveMessage = 'This is a sensitive message'
    cy.sendMessage(sensitiveMessage)
    
    // Message should be displayed normally to sender
    cy.waitForMessage(sensitiveMessage)
    
    // But should be encrypted in storage (this would need to be tested differently)
    cy.get('[data-testid="message-encrypted-indicator"]').should('be.visible')
  })
})
