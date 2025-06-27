/// <reference types="cypress" />

// ***********************************************
// This example commands.ts shows you how to
// create various custom commands and overwrite
// existing commands.
//
// For more comprehensive examples of custom
// commands please read more here:
// https://on.cypress.io/custom-commands
// ***********************************************

Cypress.Commands.add('login', (email: string, password: string) => {
  cy.visit('/login')
  cy.get('[data-testid="email-input"]').type(email)
  cy.get('[data-testid="password-input"]').type(password)
  cy.get('[data-testid="login-button"]').click()
  cy.url().should('not.include', '/login')
})

Cypress.Commands.add('logout', () => {
  cy.get('[data-testid="user-menu"]').click()
  cy.get('[data-testid="logout-button"]').click()
  cy.url().should('include', '/login')
})

Cypress.Commands.add('createChat', (name: string) => {
  cy.get('[data-testid="create-chat-button"]').click()
  cy.get('[data-testid="chat-name-input"]').type(name)
  cy.get('[data-testid="create-chat-confirm"]').click()
  cy.contains(name).should('be.visible')
})

Cypress.Commands.add('sendMessage', (message: string) => {
  cy.get('[data-testid="message-input"]').type(message)
  cy.get('[data-testid="send-button"]').click()
})

Cypress.Commands.add('waitForMessage', (message: string) => {
  cy.contains('[data-testid="message"]', message).should('be.visible')
})
