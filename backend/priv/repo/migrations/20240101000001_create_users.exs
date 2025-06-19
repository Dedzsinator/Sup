defmodule Sup.Repo.Migrations.CreateUsers do
  use Ecto.Migration

  def change do
    # Enable UUID extension
    execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")

    create table(:users, primary_key: false) do
      add(:id, :binary_id, primary_key: true, default: fragment("uuid_generate_v4()"))
      add(:email, :string, null: false)
      add(:username, :string, null: false)
      add(:password_hash, :string, null: false)
      add(:phone, :string)
      add(:avatar_url, :string)
      add(:is_online, :boolean, default: false)
      add(:last_seen, :utc_datetime)
      add(:push_token, :string)

      timestamps()
    end

    create(unique_index(:users, [:email]))
    create(unique_index(:users, [:username]))
    create(index(:users, [:is_online]))
  end
end
