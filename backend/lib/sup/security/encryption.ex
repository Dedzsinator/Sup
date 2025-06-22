defmodule Sup.Security.Encryption do
  @moduledoc """
  End-to-end encryption functionality for secure message transmission.
  Implements AES-256-GCM encryption with key derivation and secure key exchange.
  """

  alias Sup.Security.Config

  @doc """
  Generate a new encryption key pair for a user
  """
  def generate_keypair do
    # Generate Ed25519 key pair for signing and key exchange
    {public_key, private_key} = :crypto.generate_key(:ecdh, :x25519)

    %{
      public_key: Base.encode64(public_key),
      private_key: Base.encode64(private_key)
    }
  end

  @doc """
  Generate a shared secret from two key pairs
  """
  def generate_shared_secret(private_key_b64, public_key_b64) do
    try do
      private_key = Base.decode64!(private_key_b64)
      public_key = Base.decode64!(public_key_b64)

      shared_secret = :crypto.compute_key(:ecdh, public_key, private_key, :x25519)
      {:ok, Base.encode64(shared_secret)}
    rescue
      _ -> {:error, :invalid_keys}
    end
  end

  @doc """
  Encrypt a message using AES-256-GCM
  """
  def encrypt_message(message, shared_secret_b64) do
    config = Config.encryption_config()

    try do
      # Derive encryption key from shared secret
      shared_secret = Base.decode64!(shared_secret_b64)
      salt = :crypto.strong_rand_bytes(16)

      {encryption_key, _} = derive_key(shared_secret, salt, config)

      # Generate IV
      iv = :crypto.strong_rand_bytes(config.iv_length)

      # Encrypt message
      {ciphertext, tag} =
        :crypto.crypto_one_time_aead(
          config.algorithm,
          encryption_key,
          iv,
          message,
          <<>>,
          true
        )

      # Combine all components
      encrypted_data = %{
        ciphertext: Base.encode64(ciphertext),
        iv: Base.encode64(iv),
        tag: Base.encode64(tag),
        salt: Base.encode64(salt)
      }

      {:ok, encrypted_data}
    rescue
      error -> {:error, {:encryption_failed, error}}
    end
  end

  @doc """
  Decrypt a message using AES-256-GCM
  """
  def decrypt_message(encrypted_data, shared_secret_b64) do
    config = Config.encryption_config()

    try do
      # Decode components
      ciphertext = Base.decode64!(encrypted_data.ciphertext)
      iv = Base.decode64!(encrypted_data.iv)
      tag = Base.decode64!(encrypted_data.tag)
      salt = Base.decode64!(encrypted_data.salt)

      # Derive decryption key
      shared_secret = Base.decode64!(shared_secret_b64)
      {decryption_key, _} = derive_key(shared_secret, salt, config)

      # Decrypt message
      case :crypto.crypto_one_time_aead(
             config.algorithm,
             decryption_key,
             iv,
             ciphertext,
             <<>>,
             tag,
             false
           ) do
        plaintext when is_binary(plaintext) -> {:ok, plaintext}
        :error -> {:error, :decryption_failed}
      end
    rescue
      error -> {:error, {:decryption_failed, error}}
    end
  end

  @doc """
  Generate a secure session key for temporary encryption
  """
  def generate_session_key do
    # 256 bits
    key = :crypto.strong_rand_bytes(32)
    Base.encode64(key)
  end

  @doc """
  Encrypt using session key (for group messages)
  """
  def encrypt_with_session_key(message, session_key_b64) do
    config = Config.encryption_config()

    try do
      session_key = Base.decode64!(session_key_b64)
      iv = :crypto.strong_rand_bytes(config.iv_length)

      {ciphertext, tag} =
        :crypto.crypto_one_time_aead(
          config.algorithm,
          session_key,
          iv,
          message,
          <<>>,
          true
        )

      encrypted_data = %{
        ciphertext: Base.encode64(ciphertext),
        iv: Base.encode64(iv),
        tag: Base.encode64(tag)
      }

      {:ok, encrypted_data}
    rescue
      error -> {:error, {:encryption_failed, error}}
    end
  end

  @doc """
  Decrypt using session key
  """
  def decrypt_with_session_key(encrypted_data, session_key_b64) do
    config = Config.encryption_config()

    try do
      session_key = Base.decode64!(session_key_b64)
      ciphertext = Base.decode64!(encrypted_data.ciphertext)
      iv = Base.decode64!(encrypted_data.iv)
      tag = Base.decode64!(encrypted_data.tag)

      case :crypto.crypto_one_time_aead(
             config.algorithm,
             session_key,
             iv,
             ciphertext,
             <<>>,
             tag,
             false
           ) do
        plaintext when is_binary(plaintext) -> {:ok, plaintext}
        :error -> {:error, :decryption_failed}
      end
    rescue
      error -> {:error, {:decryption_failed, error}}
    end
  end

  @doc """
  Hash a password securely (for additional verification)
  """
  def hash_password(password, salt \\ nil) do
    salt = salt || :crypto.strong_rand_bytes(16)
    config = Config.encryption_config()

    hash = :crypto.pbkdf2_hmac(:sha256, password, salt, config.iterations, 32)

    %{
      hash: Base.encode64(hash),
      salt: Base.encode64(salt)
    }
  end

  @doc """
  Verify a password hash
  """
  def verify_password(password, hash_b64, salt_b64) do
    try do
      config = Config.encryption_config()
      salt = Base.decode64!(salt_b64)
      expected_hash = Base.decode64!(hash_b64)

      computed_hash = :crypto.pbkdf2_hmac(:sha256, password, salt, config.iterations, 32)

      :crypto.hash_equals(expected_hash, computed_hash)
    rescue
      _ -> false
    end
  end

  @doc """
  Generate a secure random token
  """
  def generate_secure_token(length \\ 32) do
    :crypto.strong_rand_bytes(length)
    |> Base.encode64()
    |> binary_part(0, length)
  end

  # Private functions

  defp derive_key(shared_secret, salt, config) do
    key =
      :crypto.pbkdf2_hmac(
        :sha256,
        shared_secret,
        salt,
        config.iterations,
        config.key_length
      )

    {key, salt}
  end
end
