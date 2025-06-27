defmodule Sup.Repo.Migrations.CreateDeliveryReceipts do
  use Ecto.Migration

  def change do
    create table(:delivery_receipts, primary_key: false) do
      add(:id, :binary_id, primary_key: true, default: fragment("uuid_generate_v4()"))
      add(:message_id, :binary_id, null: false)
      add(:user_id, :binary_id, null: false)
      add(:status, :string, null: false, default: "sent")
      add(:sent_at, :utc_datetime)
      add(:delivered_at, :utc_datetime)
      add(:read_at, :utc_datetime)

      timestamps()
    end

    create(unique_index(:delivery_receipts, [:message_id, :user_id]))
    create(index(:delivery_receipts, [:user_id]))
    create(index(:delivery_receipts, [:status]))
  end
end
