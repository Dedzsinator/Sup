describe('Spam Detection Integration', () => {
  beforeEach(() => {
    cy.login('test@example.com', 'password123')
    cy.createChat('Spam Detection Test')
    cy.contains('Spam Detection Test').click()
  })

  it('should allow normal messages through', () => {
    const normalMessage = 'Hello everyone, how are you doing today?'
    cy.sendMessage(normalMessage)
    cy.waitForMessage(normalMessage)
    
    // Should not show any spam warnings
    cy.get('[data-testid="spam-warning"]').should('not.exist')
  })

  it('should detect and flag obvious spam', () => {
    const spamMessage = 'WIN $1000 NOW! CLICK HERE FOR FREE MONEY! URGENT!'
    cy.sendMessage(spamMessage)
    
    // Should show spam warning
    cy.get('[data-testid="spam-warning"]').should('be.visible')
    cy.get('[data-testid="spam-confidence"]').should('contain', 'High confidence')
  })

  it('should handle spam detection service unavailable', () => {
    // Mock the spam detection service being down
    cy.intercept('POST', '**/api/spam/check', { statusCode: 500 }).as('spamServiceDown')
    
    const message = 'This message should still go through'
    cy.sendMessage(message)
    
    cy.wait('@spamServiceDown')
    cy.waitForMessage(message)
    
    // Should show fallback notice
    cy.get('[data-testid="fallback-notice"]').should('contain', 'Using fallback spam detection')
  })

  it('should allow moderators to override spam detection', () => {
    const flaggedMessage = 'This contains word: MONEY but is legitimate'
    cy.sendMessage(flaggedMessage)
    
    // Should initially be flagged
    cy.get('[data-testid="spam-warning"]').should('be.visible')
    
    // Moderator can override
    cy.get('[data-testid="override-spam"]').click()
    cy.get('[data-testid="confirm-override"]').click()
    
    // Message should now be visible normally
    cy.waitForMessage(flaggedMessage)
    cy.get('[data-testid="spam-warning"]').should('not.exist')
  })

  it('should learn from user feedback', () => {
    const borderlineMessage = 'Check out this deal on legitimate products'
    cy.sendMessage(borderlineMessage)
    
    // If flagged as spam, user can mark as not spam
    cy.get('[data-testid="not-spam-button"]').click()
    
    // Should submit feedback to improve model
    cy.get('[data-testid="feedback-submitted"]').should('be.visible')
      .and('contain', 'Thank you for your feedback')
  })
})
