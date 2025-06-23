defmodule Sup.Messaging.MessageAnalytics do
  @moduledoc """
  Schema for message analytics and metrics.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "message_analytics" do
    field(:room_id, :binary_id)
    field(:user_id, :binary_id)
    field(:date, :date)
    field(:message_count, :integer, default: 0)
    field(:active_users, :integer, default: 0)
    field(:total_characters, :integer, default: 0)
    field(:reactions_count, :integer, default: 0)
    field(:threads_count, :integer, default: 0)
    field(:media_messages_count, :integer, default: 0)
    field(:peak_concurrent_users, :integer, default: 0)
    field(:metadata, :map, default: %{})

    timestamps()
  end

  def changeset(analytics, attrs) do
    analytics
    |> cast(attrs, [
      :room_id,
      :user_id,
      :date,
      :message_count,
      :active_users,
      :total_characters,
      :reactions_count,
      :threads_count,
      :media_messages_count,
      :peak_concurrent_users,
      :metadata
    ])
    |> validate_required([:room_id, :date])
    |> validate_number(:message_count, greater_than_or_equal_to: 0)
    |> validate_number(:active_users, greater_than_or_equal_to: 0)
    |> validate_number(:total_characters, greater_than_or_equal_to: 0)
    |> validate_number(:reactions_count, greater_than_or_equal_to: 0)
    |> validate_number(:threads_count, greater_than_or_equal_to: 0)
    |> validate_number(:media_messages_count, greater_than_or_equal_to: 0)
    |> validate_number(:peak_concurrent_users, greater_than_or_equal_to: 0)
  end

  def public_fields(analytics) do
    %{
      id: analytics.id,
      room_id: analytics.room_id,
      user_id: analytics.user_id,
      date: analytics.date,
      message_count: analytics.message_count,
      active_users: analytics.active_users,
      total_characters: analytics.total_characters,
      reactions_count: analytics.reactions_count,
      threads_count: analytics.threads_count,
      media_messages_count: analytics.media_messages_count,
      peak_concurrent_users: analytics.peak_concurrent_users,
      metadata: analytics.metadata,
      created_at: analytics.inserted_at,
      updated_at: analytics.updated_at
    }
  end
end
