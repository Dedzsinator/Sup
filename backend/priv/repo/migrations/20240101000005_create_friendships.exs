defmodule Sup.Repo.Migrations.CreateFriendships do
  use Ecto.Migration

  def change do
    create table(:friendships, primary_key: false) do
      add(:id, :binary_id, primary_key: true, default: fragment("uuid_generate_v4()"))
      add(:requester_id, :binary_id, null: false)
      add(:addressee_id, :binary_id, null: false)
      # pending, accepted, blocked
      add(:status, :string, null: false, default: "pending")
      add(:created_at, :utc_datetime, null: false, default: fragment("NOW()"))
      add(:updated_at, :utc_datetime, null: false, default: fragment("NOW()"))
    end

    create(unique_index(:friendships, [:requester_id, :addressee_id]))
    create(index(:friendships, [:status]))
    create(index(:friendships, [:addressee_id]))
  end
end
