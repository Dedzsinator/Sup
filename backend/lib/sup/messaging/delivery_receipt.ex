defmodule Sup.Messaging.DeliveryReceipt do
  @moduledoc """
  Delivery receipt tracking for messages - sent/delivered/read status.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "delivery_receipts" do
    field(:message_id, :binary_id)
    field(:user_id, :binary_id)
    field(:status, Ecto.Enum, values: [:sent, :delivered, :read], default: :sent)
    field(:sent_at, :utc_datetime)
    field(:delivered_at, :utc_datetime)
    field(:read_at, :utc_datetime)

    timestamps()
  end

  def changeset(receipt, attrs) do
    receipt
    |> cast(attrs, [:message_id, :user_id, :status, :sent_at, :delivered_at, :read_at])
    |> validate_required([:message_id, :user_id, :status])
    |> unique_constraint([:message_id, :user_id])
  end
end
