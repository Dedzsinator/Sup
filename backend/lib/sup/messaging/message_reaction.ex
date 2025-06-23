defmodule Sup.Messaging.MessageReaction do
  @moduledoc """
  Message reactions (emoji responses) to messages.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "message_reactions" do
    field(:message_id, :binary_id)
    field(:user_id, :binary_id)
    field(:emoji, :string)
    field(:reaction_type, Ecto.Enum, values: [:emoji, :custom], default: :emoji)
    field(:custom_reaction_id, :binary_id)

    timestamps()
  end

  def changeset(reaction, attrs) do
    reaction
    |> cast(attrs, [:message_id, :user_id, :emoji, :reaction_type, :custom_reaction_id])
    |> validate_required([:message_id, :user_id, :emoji])
    |> validate_length(:emoji, max: 100)
    |> unique_constraint([:message_id, :user_id, :emoji])
  end
end
