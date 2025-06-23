defmodule Sup.Messaging.BotUser do
  @moduledoc """
  Schema for bot users and bot framework.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "bot_users" do
    field(:user_id, :binary_id)
    field(:bot_name, :string)
    field(:bot_type, :string)
    field(:webhook_url, :string)
    field(:api_key_hash, :string)
    field(:bot_token, :string)
    field(:configuration, :map, default: %{})
    field(:capabilities, {:array, :string}, default: [])
    field(:commands, {:array, :string}, default: [])
    field(:permissions, :map, default: %{})
    field(:rate_limit, :integer, default: 100)
    field(:rate_limit_config, :map, default: %{})
    field(:is_active, :boolean, default: true)
    field(:created_by, :binary_id)

    timestamps()
  end

  def changeset(bot_user, attrs) do
    bot_user
    |> cast(attrs, [
      :user_id,
      :bot_name,
      :bot_type,
      :webhook_url,
      :api_key_hash,
      :bot_token,
      :configuration,
      :capabilities,
      :commands,
      :permissions,
      :rate_limit,
      :rate_limit_config,
      :is_active,
      :created_by
    ])
    |> validate_required([:user_id, :bot_name, :bot_type, :created_by])
    |> validate_inclusion(:bot_type, ["webhook", "ai", "scripted", "interactive"])
    |> validate_number(:rate_limit, greater_than: 0, less_than_or_equal_to: 1000)
    |> unique_constraint(:user_id)
    |> unique_constraint(:bot_name)
  end

  def public_fields(bot_user) do
    %{
      id: bot_user.id,
      user_id: bot_user.user_id,
      bot_name: bot_user.bot_name,
      bot_type: bot_user.bot_type,
      webhook_url: bot_user.webhook_url,
      capabilities: bot_user.capabilities,
      commands: bot_user.commands,
      permissions: bot_user.permissions,
      rate_limit: bot_user.rate_limit,
      is_active: bot_user.is_active,
      created_by: bot_user.created_by,
      created_at: bot_user.inserted_at,
      updated_at: bot_user.updated_at
    }
  end
end
