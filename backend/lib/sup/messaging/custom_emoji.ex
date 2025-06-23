defmodule Sup.Messaging.CustomEmoji do
  @moduledoc """
  Schema for custom emojis in rooms.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "custom_emojis" do
    field(:name, :string)
    field(:display_name, :string)
    field(:room_id, :binary_id)
    field(:file_path, :string)
    field(:content_type, :string)
    field(:file_size, :integer)
    field(:created_by, :binary_id)
    field(:description, :string)
    field(:tags, {:array, :string}, default: [])
    field(:is_active, :boolean, default: true)
    field(:usage_count, :integer, default: 0)

    timestamps()
  end

  def changeset(custom_emoji, attrs) do
    custom_emoji
    |> cast(attrs, [
      :name,
      :display_name,
      :room_id,
      :file_path,
      :content_type,
      :file_size,
      :created_by,
      :description,
      :tags,
      :is_active,
      :usage_count
    ])
    |> validate_required([:name, :file_path, :content_type, :created_by])
    |> validate_length(:name, min: 2, max: 50)
    |> validate_format(:name, ~r/^[a-zA-Z0-9_]+$/,
      message: "can only contain letters, numbers, and underscores"
    )
    |> validate_number(:file_size, greater_than: 0)
    |> validate_inclusion(:content_type, ["image/png", "image/gif", "image/jpeg", "image/webp"])
    |> unique_constraint([:name, :room_id])
  end

  def public_fields(custom_emoji) do
    %{
      id: custom_emoji.id,
      name: custom_emoji.name,
      display_name: custom_emoji.display_name || custom_emoji.name,
      room_id: custom_emoji.room_id,
      file_path: custom_emoji.file_path,
      content_type: custom_emoji.content_type,
      file_size: custom_emoji.file_size,
      created_by: custom_emoji.created_by,
      description: custom_emoji.description,
      tags: custom_emoji.tags,
      is_active: custom_emoji.is_active,
      usage_count: custom_emoji.usage_count,
      created_at: custom_emoji.inserted_at
    }
  end
end
