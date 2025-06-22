defmodule Sup.Repo.Migrations.AddSecurityFieldsToUsers do
  use Ecto.Migration

  def change do
    alter table(:users) do
      add(:role, :string, default: "user")
      add(:account_locked, :boolean, default: false)
      add(:locked_until, :utc_datetime)
      add(:failed_login_attempts, :integer, default: 0)
      add(:public_key, :text)
      add(:private_key_hash, :string)
      add(:session_id, :string)
      add(:refresh_token_hash, :string)
    end

    create(index(:users, [:role]))
    create(index(:users, [:account_locked]))
    create(index(:users, [:session_id]))
  end
end
