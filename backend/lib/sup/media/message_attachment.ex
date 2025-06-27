defmodule Sup.Media.MessageAttachment do
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "message_attachments" do
    field(:filename, :string)
    field(:content_type, :string)
    field(:file_size, :integer)
    field(:file_path, :string)
    field(:thumbnail_path, :string)
    field(:metadata, :map, default: %{})

    belongs_to(:message, Sup.Messaging.Message)
    belongs_to(:uploaded_by, Sup.Auth.User)

    timestamps()
  end

  def changeset(attachment, attrs) do
    attachment
    |> cast(attrs, [
      :filename,
      :content_type,
      :file_size,
      :file_path,
      :thumbnail_path,
      :metadata,
      :message_id,
      :uploaded_by_id
    ])
    |> validate_required([:filename, :content_type, :file_size, :file_path])
    |> validate_number(:file_size, greater_than: 0)
    |> foreign_key_constraint(:message_id)
    |> foreign_key_constraint(:uploaded_by_id)
  end

  def public_fields(attachment) do
    %{
      id: attachment.id,
      filename: attachment.filename,
      content_type: attachment.content_type,
      file_size: attachment.file_size,
      thumbnail_path: attachment.thumbnail_path,
      metadata: attachment.metadata,
      uploaded_by_id: attachment.uploaded_by_id,
      inserted_at: attachment.inserted_at
    }
  end
end
