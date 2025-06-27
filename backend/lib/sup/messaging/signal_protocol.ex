defmodule Sup.Messaging.SignalProtocol do
  @moduledoc """
  Signal Protocol implementation for end-to-end encryption in messaging.
  Provides forward secrecy, deniability, and perfect forward secrecy.
  """

  alias Sup.Crypto.{IdentityKey, PreKey, SignedPreKey, Session}
  alias Sup.Messaging.KeyBundle
  alias Sup.Repo
  import Ecto.Query

  require Logger

  @prekey_count 100
  # @signed_prekey_lifetime_days removed as it was unused

  # Key Generation
  def generate_identity_key(user_id) do
    identity_keypair = :crypto.generate_key(:ecdh, :x25519)

    identity_key = %{
      user_id: user_id,
      public_key: elem(identity_keypair, 0),
      private_key: elem(identity_keypair, 1),
      created_at: DateTime.utc_now()
    }

    IdentityKey.changeset(%IdentityKey{}, identity_key)
    |> Repo.insert()
  end

  def generate_prekeys(user_id, count \\ @prekey_count) do
    Enum.map(1..count, fn id ->
      keypair = :crypto.generate_key(:ecdh, :x25519)

      %{
        user_id: user_id,
        key_id: id,
        public_key: elem(keypair, 0),
        private_key: elem(keypair, 1),
        created_at: DateTime.utc_now()
      }
    end)
    |> then(&Repo.insert_all(PreKey, &1))
  end

  def generate_signed_prekey(user_id) do
    identity_key = get_identity_key(user_id)
    keypair = :crypto.generate_key(:ecdh, :x25519)

    # Sign the public key with identity key
    signature = :crypto.sign(:ecdsa, :sha256, elem(keypair, 0), identity_key.private_key)

    signed_prekey = %{
      user_id: user_id,
      key_id: :crypto.strong_rand_bytes(4) |> :binary.decode_unsigned(),
      public_key: elem(keypair, 0),
      private_key: elem(keypair, 1),
      signature: signature,
      created_at: DateTime.utc_now()
    }

    SignedPreKey.changeset(%SignedPreKey{}, signed_prekey)
    |> Repo.insert()
  end

  # Key Bundle Management
  def get_key_bundle(user_id) do
    identity_key = get_identity_key(user_id)
    signed_prekey = get_current_signed_prekey(user_id)
    prekey = get_random_prekey(user_id)

    %KeyBundle{
      user_id: user_id,
      identity_key: identity_key.public_key,
      signed_prekey_id: signed_prekey.key_id,
      signed_prekey: signed_prekey.public_key,
      signed_prekey_signature: signed_prekey.signature,
      prekey_id: prekey.key_id,
      prekey: prekey.public_key
    }
  end

  def refresh_prekeys(user_id) do
    # Remove old prekeys
    from(p in PreKey, where: p.user_id == ^user_id)
    |> Repo.delete_all()

    # Generate new ones
    generate_prekeys(user_id)
  end

  def refresh_signed_prekey(user_id) do
    # Mark old signed prekey as expired
    from(sp in SignedPreKey,
      where: sp.user_id == ^user_id and is_nil(sp.expires_at)
    )
    |> Repo.update_all(set: [expires_at: DateTime.utc_now()])

    # Generate new signed prekey
    generate_signed_prekey(user_id)
  end

  # Session Management
  def establish_session(sender_id, recipient_id, key_bundle) do
    # X3DH key agreement protocol
    identity_key_sender = get_identity_key(sender_id)
    ephemeral_keypair = :crypto.generate_key(:ecdh, :x25519)

    # Perform X3DH key exchange
    dh1 =
      :crypto.compute_key(
        :ecdh,
        key_bundle.signed_prekey,
        identity_key_sender.private_key,
        :x25519
      )

    dh2 = :crypto.compute_key(:ecdh, key_bundle.identity_key, elem(ephemeral_keypair, 1), :x25519)

    dh3 =
      :crypto.compute_key(:ecdh, key_bundle.signed_prekey, elem(ephemeral_keypair, 1), :x25519)

    # Optional one-time prekey DH
    dh4 =
      if key_bundle.prekey do
        :crypto.compute_key(:ecdh, key_bundle.prekey, elem(ephemeral_keypair, 1), :x25519)
      else
        <<>>
      end

    # Derive initial root key
    kdf_input = dh1 <> dh2 <> dh3 <> dh4
    root_key = :crypto.hash(:sha256, kdf_input)

    # Initialize Double Ratchet
    session = %{
      sender_id: sender_id,
      recipient_id: recipient_id,
      root_key: root_key,
      chain_key_send: :crypto.hash(:sha256, "send" <> root_key),
      chain_key_recv: :crypto.hash(:sha256, "recv" <> root_key),
      message_number_send: 0,
      message_number_recv: 0,
      ephemeral_public: elem(ephemeral_keypair, 0),
      ephemeral_private: elem(ephemeral_keypair, 1),
      created_at: DateTime.utc_now()
    }

    Session.changeset(%Session{}, session)
    |> Repo.insert()
  end

  # Message Encryption/Decryption
  def encrypt_message(sender_id, recipient_id, plaintext) do
    session = get_session(sender_id, recipient_id)

    if session do
      # Generate message key from chain key
      message_key =
        :crypto.hash(:sha256, session.chain_key_send <> <<session.message_number_send::32>>)

      # Encrypt message using AES-256-GCM
      {ciphertext, tag} =
        :crypto.crypto_one_time_aead(
          :aes_256_gcm,
          message_key,
          :crypto.strong_rand_bytes(12),
          plaintext,
          <<>>,
          true
        )

      encrypted_message = %{
        sender_id: sender_id,
        recipient_id: recipient_id,
        session_id: session.id,
        message_number: session.message_number_send,
        ciphertext: ciphertext,
        auth_tag: tag,
        created_at: DateTime.utc_now()
      }

      # Update session state
      new_chain_key = :crypto.hash(:sha256, session.chain_key_send)

      Session.changeset(session, %{
        chain_key_send: new_chain_key,
        message_number_send: session.message_number_send + 1
      })
      |> Repo.update()

      {:ok, encrypted_message}
    else
      {:error, "no_session"}
    end
  end

  def decrypt_message(encrypted_message) do
    session = Repo.get(Session, encrypted_message.session_id)

    if session do
      # Generate message key
      message_key =
        :crypto.hash(:sha256, session.chain_key_recv <> <<encrypted_message.message_number::32>>)

      # Decrypt message
      case :crypto.crypto_one_time_aead(
             :aes_256_gcm,
             message_key,
             :crypto.strong_rand_bytes(12),
             encrypted_message.ciphertext,
             <<>>,
             encrypted_message.auth_tag,
             false
           ) do
        plaintext when is_binary(plaintext) ->
          # Update session state
          new_chain_key = :crypto.hash(:sha256, session.chain_key_recv)

          Session.changeset(session, %{
            chain_key_recv: new_chain_key,
            message_number_recv: encrypted_message.message_number + 1
          })
          |> Repo.update()

          {:ok, plaintext}

        :error ->
          {:error, "decryption_failed"}
      end
    else
      {:error, "session_not_found"}
    end
  end

  # Double Ratchet Protocol
  def perform_dh_ratchet(session_id, new_public_key) do
    session = Repo.get(Session, session_id)

    if session do
      # Generate new ephemeral keypair
      new_keypair = :crypto.generate_key(:ecdh, :x25519)

      # Perform DH with received public key
      dh_output = :crypto.compute_key(:ecdh, new_public_key, session.ephemeral_private, :x25519)

      # Update root key and chain keys
      new_root_key = :crypto.hash(:sha256, session.root_key <> dh_output)
      new_chain_key_send = :crypto.hash(:sha256, "send" <> new_root_key)
      new_chain_key_recv = :crypto.hash(:sha256, "recv" <> new_root_key)

      Session.changeset(session, %{
        root_key: new_root_key,
        chain_key_send: new_chain_key_send,
        chain_key_recv: new_chain_key_recv,
        ephemeral_public: elem(new_keypair, 0),
        ephemeral_private: elem(new_keypair, 1),
        message_number_send: 0,
        message_number_recv: 0
      })
      |> Repo.update()
    else
      {:error, "session_not_found"}
    end
  end

  # Helper functions
  defp get_identity_key(user_id) do
    from(ik in IdentityKey, where: ik.user_id == ^user_id)
    |> Repo.one()
  end

  defp get_current_signed_prekey(user_id) do
    from(sp in SignedPreKey,
      where: sp.user_id == ^user_id and is_nil(sp.expires_at),
      order_by: [desc: sp.created_at],
      limit: 1
    )
    |> Repo.one()
  end

  defp get_random_prekey(user_id) do
    # Get a random unused prekey
    from(p in PreKey,
      where: p.user_id == ^user_id and is_nil(p.used_at),
      order_by: fragment("RANDOM()"),
      limit: 1
    )
    |> Repo.one()
  end

  defp get_session(sender_id, recipient_id) do
    from(s in Session,
      where: s.sender_id == ^sender_id and s.recipient_id == ^recipient_id,
      order_by: [desc: s.created_at],
      limit: 1
    )
    |> Repo.one()
  end
end
