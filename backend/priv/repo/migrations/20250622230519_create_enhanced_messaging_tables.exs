defmodule Sup.Repo.Migrations.CreateEnhancedMessagingTables do
  use Ecto.Migration

  def change do
    # Message threads table for threaded conversations
    create table(:message_threads, primary_key: false) do
      add(:id, :binary_id, primary_key: true)
      add(:room_id, references(:rooms, type: :binary_id, on_delete: :delete_all), null: false)
      add(:parent_message_id, :binary_id, null: false)
      add(:created_by, references(:users, type: :binary_id, on_delete: :delete_all), null: false)
      add(:message_count, :integer, default: 1)
      add(:participants, {:array, :binary_id}, default: [])
      add(:last_message_id, :binary_id)
      add(:last_activity_at, :utc_datetime)
      add(:is_pinned, :boolean, default: false)
      add(:metadata, :map, default: %{})

      timestamps()
    end

    create(index(:message_threads, [:room_id]))
    create(index(:message_threads, [:parent_message_id]))
    create(index(:message_threads, [:created_by]))
    create(index(:message_threads, [:last_activity_at]))

    # Message reactions table
    create table(:message_reactions, primary_key: false) do
      add(:id, :binary_id, primary_key: true)
      add(:message_id, :binary_id, null: false)
      add(:user_id, references(:users, type: :binary_id, on_delete: :delete_all), null: false)
      add(:emoji, :string, null: false)
      add(:reaction_type, :string, default: "emoji")
      add(:custom_reaction_id, :binary_id)

      timestamps()
    end

    create(unique_index(:message_reactions, [:message_id, :user_id, :emoji]))
    create(index(:message_reactions, [:message_id]))
    create(index(:message_reactions, [:user_id]))

    # Enhanced delivery receipts with thread support
    alter table(:delivery_receipts) do
      add(:thread_id, :binary_id)
      add(:metadata, :map, default: %{})
    end

    create(index(:delivery_receipts, [:thread_id]))

    # Enhanced messages schema updates (for ScyllaDB compatibility tracking)
    create table(:message_metadata, primary_key: false) do
      add(:id, :binary_id, primary_key: true)
      add(:message_id, :binary_id, null: false)
      add(:mentions, {:array, :string}, default: [])
      add(:hashtags, {:array, :string}, default: [])
      add(:links, {:array, :string}, default: [])
      add(:media_metadata, :map, default: %{})
      add(:location, :map)
      add(:quoted_message_id, :binary_id)
      # For full-text search
      add(:search_vector, :string)
      # For deduplication
      add(:content_hash, :string)

      timestamps()
    end

    create(unique_index(:message_metadata, [:message_id]))
    create(index(:message_metadata, [:mentions], using: :gin))
    create(index(:message_metadata, [:hashtags], using: :gin))

    # Offline message queue
    create table(:offline_messages, primary_key: false) do
      add(:id, :binary_id, primary_key: true)
      add(:user_id, references(:users, type: :binary_id, on_delete: :delete_all), null: false)
      add(:message_id, :binary_id, null: false)
      add(:room_id, :binary_id, null: false)
      add(:priority, :integer, default: 1)
      add(:attempts, :integer, default: 0)
      add(:max_attempts, :integer, default: 3)
      add(:scheduled_for, :utc_datetime)
      add(:delivered, :boolean, default: false)
      add(:metadata, :map, default: %{})

      timestamps()
    end

    create(index(:offline_messages, [:user_id, :delivered]))
    create(index(:offline_messages, [:scheduled_for]))
    create(index(:offline_messages, [:room_id]))

    # Bot framework tables
    create table(:bot_users, primary_key: false) do
      add(:id, :binary_id, primary_key: true)
      add(:user_id, references(:users, type: :binary_id, on_delete: :delete_all), null: false)
      add(:bot_name, :string, null: false)
      # webhook, ai, scripted
      add(:bot_type, :string, null: false)
      add(:webhook_url, :string)
      add(:api_key_hash, :string)
      add(:configuration, :map, default: %{})
      add(:capabilities, {:array, :string}, default: [])
      add(:is_active, :boolean, default: true)
      # requests per minute
      add(:rate_limit, :integer, default: 100)

      timestamps()
    end

    create(unique_index(:bot_users, [:user_id]))
    create(index(:bot_users, [:bot_name]))
    create(index(:bot_users, [:is_active]))

    # Analytics and metrics
    create table(:message_analytics, primary_key: false) do
      add(:id, :binary_id, primary_key: true)
      add(:room_id, :binary_id, null: false)
      add(:user_id, :binary_id)
      add(:date, :date, null: false)
      add(:message_count, :integer, default: 0)
      add(:active_users, :integer, default: 0)
      add(:total_characters, :integer, default: 0)
      add(:reactions_count, :integer, default: 0)
      add(:threads_count, :integer, default: 0)
      add(:media_messages_count, :integer, default: 0)
      add(:peak_concurrent_users, :integer, default: 0)
      add(:metadata, :map, default: %{})

      timestamps()
    end

    create(unique_index(:message_analytics, [:room_id, :date]))
    create(index(:message_analytics, [:date]))
    create(index(:message_analytics, [:user_id, :date]))

    # Custom emoji and stickers
    create table(:custom_emojis, primary_key: false) do
      add(:id, :binary_id, primary_key: true)
      add(:name, :string, null: false)
      add(:image_url, :string, null: false)
      add(:created_by, references(:users, type: :binary_id, on_delete: :delete_all), null: false)
      # null for global emojis
      add(:room_id, :binary_id)
      add(:is_animated, :boolean, default: false)
      add(:tags, {:array, :string}, default: [])
      add(:usage_count, :integer, default: 0)
      add(:is_approved, :boolean, default: false)

      timestamps()
    end

    create(unique_index(:custom_emojis, [:name, :room_id]))
    create(index(:custom_emojis, [:created_by]))
    create(index(:custom_emojis, [:room_id]))
    create(index(:custom_emojis, [:is_approved]))

    # Multi-device sync state
    create table(:device_sync_state, primary_key: false) do
      add(:id, :binary_id, primary_key: true)
      add(:user_id, references(:users, type: :binary_id, on_delete: :delete_all), null: false)
      add(:device_id, :string, null: false)
      # web, mobile, desktop
      add(:device_type, :string, null: false)
      add(:last_sync_timestamp, :utc_datetime)
      add(:sync_state, :map, default: %{})
      add(:unread_counts, :map, default: %{})
      add(:is_active, :boolean, default: true)

      timestamps()
    end

    create(unique_index(:device_sync_state, [:user_id, :device_id]))
    create(index(:device_sync_state, [:user_id, :is_active]))

    # Rich media attachments
    create table(:message_attachments, primary_key: false) do
      add(:id, :binary_id, primary_key: true)
      add(:message_id, :binary_id, null: false)
      add(:file_name, :string, null: false)
      add(:file_type, :string, null: false)
      add(:file_size, :integer, null: false)
      add(:file_url, :string, null: false)
      add(:thumbnail_url, :string)
      # for audio/video
      add(:duration, :integer)
      # for images/videos
      add(:dimensions, :map)
      add(:metadata, :map, default: %{})
      add(:is_encrypted, :boolean, default: false)
      add(:encryption_key, :binary)

      timestamps()
    end

    create(index(:message_attachments, [:message_id]))
    create(index(:message_attachments, [:file_type]))
  end
end
