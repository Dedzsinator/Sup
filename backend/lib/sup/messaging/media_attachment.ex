defmodule Sup.Messaging.MediaAttachment do
  @moduledoc """
  Schema for media attachments associated with messages.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "message_attachments" do
    field(:message_id, :binary_id)
    field(:original_filename, :string)
    field(:stored_filename, :string)
    field(:file_name, :string)
    field(:file_type, :string)
    field(:file_size, :integer)
    field(:file_url, :string)
    field(:file_path, :string)
    field(:media_type, :string)
    field(:thumbnail_url, :string)
    field(:duration, :integer)
    field(:dimensions, :map)
    field(:metadata, :map, default: %{})
    field(:uploaded_by, :binary_id)
    field(:uploaded_at, :utc_datetime)
    field(:is_encrypted, :boolean, default: false)
    field(:encryption_key, :binary)

    timestamps()
  end

  def changeset(attachment, attrs) do
    attachment
    |> cast(attrs, [
      :message_id,
      :original_filename,
      :stored_filename,
      :file_name,
      :file_type,
      :file_size,
      :file_url,
      :file_path,
      :media_type,
      :thumbnail_url,
      :duration,
      :dimensions,
      :metadata,
      :uploaded_by,
      :uploaded_at,
      :is_encrypted,
      :encryption_key
    ])
    |> validate_required([:file_name, :file_type, :file_size, :file_url, :uploaded_by])
    |> validate_inclusion(:media_type, ["image", "video", "audio", "document"])
    |> validate_number(:file_size, greater_than: 0)
    |> validate_number(:duration, greater_than_or_equal_to: 0)
  end

  def public_fields(attachment) do
    %{
      id: attachment.id,
      message_id: attachment.message_id,
      original_filename: attachment.original_filename,
      file_name: attachment.file_name,
      file_type: attachment.file_type,
      file_size: attachment.file_size,
      file_url: attachment.file_url,
      media_type: attachment.media_type,
      thumbnail_url: attachment.thumbnail_url,
      duration: attachment.duration,
      dimensions: attachment.dimensions,
      metadata: attachment.metadata,
      uploaded_by: attachment.uploaded_by,
      uploaded_at: attachment.uploaded_at,
      is_encrypted: attachment.is_encrypted,
      created_at: attachment.inserted_at
    }
  end
end
