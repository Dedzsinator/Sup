describe('Authentication Flow', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('should redirect to login when not authenticated', () => {
    cy.url().should('include', '/login')
    cy.contains('Sign in to Sup').should('be.visible')
  })

  it('should allow user to login with valid credentials', () => {
    cy.get('[data-testid="email-input"]').type('test@example.com')
    cy.get('[data-testid="password-input"]').type('password123')
    cy.get('[data-testid="login-button"]').click()
    
    // Should redirect to main app
    cy.url().should('not.include', '/login')
    cy.get('[data-testid="main-content"]').should('be.visible')
  })

  it('should show error message for invalid credentials', () => {
    cy.get('[data-testid="email-input"]').type('invalid@example.com')
    cy.get('[data-testid="password-input"]').type('wrongpassword')
    cy.get('[data-testid="login-button"]').click()
    
    cy.get('[data-testid="error-message"]').should('contain', 'Invalid credentials')
  })

  it('should allow user to register new account', () => {
    cy.get('[data-testid="register-link"]').click()
    cy.url().should('include', '/register')
    
    cy.get('[data-testid="username-input"]').type('newuser')
    cy.get('[data-testid="email-input"]').type('newuser@example.com')
    cy.get('[data-testid="password-input"]').type('password123')
    cy.get('[data-testid="confirm-password-input"]').type('password123')
    cy.get('[data-testid="register-button"]').click()
    
    // Should redirect to verification or main app
    cy.url().should('not.include', '/register')
  })
})
