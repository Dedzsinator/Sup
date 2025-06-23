defmodule Sup.Repo.Migrations.CreateSignalProtocolTables do
  use Ecto.Migration

  def change do
    # Identity keys table
    create table(:identity_keys, primary_key: false) do
      add(:id, :binary_id, primary_key: true)
      add(:user_id, references(:users, type: :binary_id, on_delete: :delete_all), null: false)
      add(:public_key, :binary, null: false)
      add(:private_key, :binary, null: false)
      
      timestamps()
    end
    
    create(unique_index(:identity_keys, [:user_id]))
    
    # One-time prekeys table
    create table(:prekeys, primary_key: false) do
      add(:id, :binary_id, primary_key: true)
      add(:user_id, references(:users, type: :binary_id, on_delete: :delete_all), null: false)
      add(:key_id, :integer, null: false)
      add(:public_key, :binary, null: false)
      add(:private_key, :binary, null: false)
      add(:used_at, :utc_datetime)
      
      timestamps()
    end
    
    create(unique_index(:prekeys, [:user_id, :key_id]))
    create(index(:prekeys, [:user_id, :used_at]))
    
    # Signed prekeys table
    create table(:signed_prekeys, primary_key: false) do
      add(:id, :binary_id, primary_key: true)
      add(:user_id, references(:users, type: :binary_id, on_delete: :delete_all), null: false)
      add(:key_id, :integer, null: false)
      add(:public_key, :binary, null: false)
      add(:private_key, :binary, null: false)
      add(:signature, :binary, null: false)
      add(:expires_at, :utc_datetime)
      
      timestamps()
    end
    
    create(unique_index(:signed_prekeys, [:user_id, :key_id]))
    create(index(:signed_prekeys, [:user_id, :expires_at]))
    
    # Crypto sessions table for Double Ratchet
    create table(:crypto_sessions, primary_key: false) do
      add(:id, :binary_id, primary_key: true)
      add(:sender_id, references(:users, type: :binary_id, on_delete: :delete_all), null: false)
      add(:recipient_id, references(:users, type: :binary_id, on_delete: :delete_all), null: false)
      add(:root_key, :binary, null: false)
      add(:chain_key_send, :binary, null: false)
      add(:chain_key_recv, :binary, null: false)
      add(:message_number_send, :integer, default: 0)
      add(:message_number_recv, :integer, default: 0)
      add(:ephemeral_public, :binary)
      add(:ephemeral_private, :binary)
      add(:previous_counter, :integer, default: 0)
      
      timestamps()
    end
    
    create(unique_index(:crypto_sessions, [:sender_id, :recipient_id]))
    create(index(:crypto_sessions, [:recipient_id, :sender_id]))
    
    # Encrypted messages table
    create table(:encrypted_messages, primary_key: false) do
      add(:id, :binary_id, primary_key: true)
      add(:sender_id, references(:users, type: :binary_id, on_delete: :delete_all), null: false)
      add(:recipient_id, references(:users, type: :binary_id, on_delete: :delete_all), null: false)
      add(:session_id, references(:crypto_sessions, type: :binary_id, on_delete: :delete_all), null: false)
      add(:message_number, :integer, null: false)
      add(:ciphertext, :binary, null: false)
      add(:auth_tag, :binary, null: false)
      add(:ephemeral_key, :binary)
      add(:decrypted, :boolean, default: false)
      
      timestamps()
    end
    
    create(index(:encrypted_messages, [:sender_id, :recipient_id]))
    create(index(:encrypted_messages, [:session_id, :message_number]))
    create(index(:encrypted_messages, [:recipient_id, :decrypted]))
  end
end
